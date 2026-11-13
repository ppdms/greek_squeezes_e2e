"""Registration of a squeeze's second view (Rotation2) onto its first (Rotation1).

Every squeeze has two photographs of the same stone. A `Registration` maps Rotation2 pixel coordinates to
Rotation1 pixel coordinates; `.inverse()` maps back. The detector stores one per squeeze (`det/{base}.json` carries
`registration` = `Registration.to_dict()`), and the relief and crop stages rebuild it with
`Registration.from_dict(record['registration'])` to cut crops from the second view.

Fitting is done in two steps. `register_pair` first aligns the images (phase correlation, ECC, optionally local ECC
patches), then, once letters were detected in both views, refines the fit on the detected letter centres (trimmed
ICP, piecewise seam models, thin-plate-spline correction). `fit_centres` fits centres only; training uses it on the
annotated letters.

Coordinates are (x, y) in original image pixels. `inlier_fraction` is the share of letter centres landing within a
quarter letter height of a reference centre; it is None for image-only fits (their quality is an ECC/phase proxy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from scipy.linalg import solve
from scipy.spatial import cKDTree

REG_VERSION = 'reg_v2'
FIT_VERSION = 'reg_v2_orientation_checked_v1'
LOCAL_ECC_THRESHOLD = .70


def _points(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('points must be a finite N x 2 array')
    return p


def _apply(matrix, points):
    return points @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]


def _homogeneous(matrix):
    return np.vstack([matrix, [0., 0., 1.]])


def _scale_frame(scale):
    # cv2.resize maps pixel centres: x_small = scale * (x + .5) - .5.
    return np.array([[scale, 0., (scale - 1) / 2], [0., scale, (scale - 1) / 2], [0., 0., 1.]])


@dataclass
class Registration:
    """A serializable global similarity, or a piecewise map built from local transforms.

    `model` is 'similarity' or 'piecewise'. A piecewise map is one of: `grid` (3x3 image-ECC patches, bilinearly
    blended), `axis`/`knots`/`matrices` (similarities blended along one image axis) or `tps` (a thin-plate-spline
    correction on top of `matrix`). Centre quality is pass (>= .98), usable (>= .9) or late_fusion (< .9);
    image-only quality is image_usable / image_poor.
    """

    model: Literal['similarity', 'piecewise']
    parameters: dict
    inlier_fraction: float | None = None
    quality: str = 'unmeasured'
    version: str = REG_VERSION
    diagnostics: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record: dict) -> 'Registration':
        """Rebuild from `to_dict()`, e.g. the `registration` entry of a detection record."""
        return cls(**record)

    def to_dict(self) -> dict:
        return {'model': self.model, 'parameters': self.parameters, 'inlier_fraction': self.inlier_fraction,
                'quality': self.quality, 'version': self.version, 'diagnostics': self.diagnostics}

    def apply(self, points):
        """Map (N, 2) points; for an `inverse()` of a piecewise map, solve the forward map by damped Newton."""
        p = _points(points)
        if self.parameters.get('inverse'):
            forward = Registration(self.model, {k: v for k, v in self.parameters.items() if k != 'inverse'})
            # Solve the actual blended map, rather than blending inverses.
            # Undamped Newton can oscillate across seam intervals even when
            # every interval is strictly monotone. Backtrack each point until
            # its forward residual decreases; never return an approximate map.
            x = _apply(np.linalg.inv(_homogeneous(self.parameters['matrix']))[:2], p)
            active = np.arange(len(p))
            for _ in range(40):
                current = x[active]
                fx = forward.apply(current)
                residual = fx - p[active]
                error = np.max(np.abs(residual), axis=1)
                pending = error >= 1e-7
                if not pending.any():
                    return x
                active = active[pending]
                current, fx, residual, error = current[pending], fx[pending], residual[pending], error[pending]
                jacobian = _jacobian(forward, current, fx)
                try:
                    step = np.linalg.solve(jacobian, residual[..., None])[..., 0]
                except np.linalg.LinAlgError as exc:
                    raise ValueError('piecewise registration is not invertible here') from exc
                remaining = np.arange(len(active))
                damping = 1.
                for _ in range(20):
                    candidate = current[remaining] - damping * step[remaining]
                    candidate_error = np.max(np.abs(forward.apply(candidate) - p[active[remaining]]), axis=1)
                    improved = candidate_error < error[remaining]
                    x[active[remaining[improved]]] = candidate[improved]
                    remaining = remaining[~improved]
                    if not len(remaining):
                        break
                    damping *= .5
                if len(remaining):
                    raise ValueError('piecewise inverse did not converge')
            if not len(active) or np.max(np.abs(forward.apply(x[active]) - p[active])) < 1e-7:
                return x
            raise ValueError('piecewise inverse did not converge')
        if self.model == 'similarity':
            return _apply(self.parameters['matrix'], p)
        if 'tps' in self.parameters:
            tps = self.parameters['tps']
            x = (p - tps['origin']) / tps['scale']
            controls = np.asarray(tps['controls']).reshape(-1, 2)
            displacement = _thin_plate(x, controls) @ np.asarray(tps['weights']).reshape(-1, 2)
            displacement += np.column_stack([np.ones(len(p)), x]) @ np.asarray(tps['affine'])
            return _apply(self.parameters['matrix'], p) + displacement
        if 'grid' in self.parameters:
            grid = self.parameters['grid']
            frame = _apply(grid['frame'], p)
            ix, wx = _interval(grid['x_knots'], frame[:, 0])
            iy, wy = _interval(grid['y_knots'], frame[:, 1])
            matrices = np.asarray(grid['matrices'])
            result = np.zeros_like(p)
            for dy, weight_y in ((0, 1 - wy), (1, wy)):
                for dx, weight_x in ((0, 1 - wx), (1, wx)):
                    local = matrices[iy + dy, ix + dx]
                    transformed = np.einsum('nij,nj->ni', local[:, :, :2], p) + local[:, :, 2]
                    result += (weight_x * weight_y)[:, None] * transformed
            return result
        axis = self.parameters['axis']
        knots = np.asarray(self.parameters['knots'])
        matrices = np.asarray(self.parameters['matrices'])
        if not len(p):
            return p.copy()
        upper = np.clip(np.searchsorted(knots, p[:, axis]), 1, len(knots) - 1)
        lower = upper - 1
        weight = np.clip((p[:, axis] - knots[lower]) / (knots[upper] - knots[lower]), 0, 1)
        a = np.einsum('nij,nj->ni', matrices[lower, :, :2], p) + matrices[lower, :, 2]
        b = np.einsum('nij,nj->ni', matrices[upper, :, :2], p) + matrices[upper, :, 2]
        return (1 - weight[:, None]) * a + weight[:, None] * b

    def inverse(self) -> 'Registration':
        """Rotation1 -> Rotation2 (same record layout; a piecewise inverse is solved on demand by `apply`)."""
        params = dict(self.parameters)
        if self.model == 'similarity':
            params['matrix'] = np.linalg.inv(_homogeneous(params['matrix']))[:2].tolist()
        elif params.pop('inverse', False) is False:
            params['inverse'] = True
        return Registration(self.model, params, self.inlier_fraction, self.quality, self.version, self.diagnostics)


def _similarity(source, target):
    ma, mb = source.mean(0), target.mean(0)
    a, b = source - ma, target - mb
    u, singular, vt = np.linalg.svd(a.T @ b)
    signs = np.ones(2)
    if np.linalg.det(vt.T @ u.T) < 0:
        signs[-1] = -1
    rotation = vt.T @ np.diag(signs) @ u.T
    scale = (singular * signs).sum() / max(np.square(a).sum(), 1e-12)
    return np.column_stack([scale * rotation, mb - scale * rotation @ ma])


def _interval(knots, values):
    knots = np.asarray(knots)
    lower = np.clip(np.searchsorted(knots, values) - 1, 0, len(knots) - 2)
    weight = np.clip((values - knots[lower]) / (knots[lower + 1] - knots[lower]), 0, 1)
    return lower, weight


def _quality(fraction):
    return 'pass' if fraction >= .98 else 'usable' if fraction >= .9 else 'late_fusion'


def _thin_plate(x, controls):
    squared = np.square(x[:, None, :] - controls[None, :, :]).sum(axis=2)
    return .5 * squared * np.log(np.maximum(squared, 1e-30))


def _tps_parameters(points, displacement, h, regularization):
    origin = points.mean(0)
    x = (points - origin) / h
    polynomial = np.column_stack([np.ones(len(x)), x])
    kernel = _thin_plate(x, x)
    kernel.flat[::len(x) + 1] += regularization
    system = np.block([[kernel, polynomial], [polynomial.T, np.zeros((3, 3))]])
    coefficients = solve(system, np.vstack([displacement, np.zeros((3, 2))]), assume_a='sym')
    return {'origin': origin.tolist(), 'scale': h, 'regularization': regularization, 'controls': x.tolist(),
            'weights': coefficients[:-3].tolist(), 'affine': coefficients[-3:].tolist()}


def _source_probes(points, h, source_size=None):
    """Check source-pixel orientation at <= one tenth of a letter height.

    This is a sampled fold rejection, not a proof of global bijectivity. The
    actual inverse still has to converge at every requested warp/crop point.
    """
    low, high = ((np.zeros(2), np.asarray(source_size, float) - 1) if source_size is not None
                 else (points.min(0) - 2 * h, points.max(0) + 2 * h))
    count = np.maximum(2, np.ceil((high - low) / (h / 10)).astype(int) + 1)
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], count[0]),
                         np.linspace(low[1], high[1], count[1]))
    return np.concatenate([points, np.column_stack([xx.ravel(), yy.ravel()])])


def _jacobian(reg, points, value=None):
    """Exact TPS derivatives avoid two repeated radial-basis evaluations."""
    tps = reg.parameters.get('tps')
    if tps is not None:
        controls, weights = np.asarray(tps['controls']), np.asarray(tps['weights'])
        affine = np.asarray(reg.parameters['matrix'])[:, :2] + np.asarray(tps['affine'])[1:].T / tps['scale']
        delta = ((points - tps['origin']) / tps['scale'])[:, None, :] - controls[None, :, :]
        squared = np.einsum('nki,nki->nk', delta, delta)
        delta *= (np.log(np.maximum(squared, 1e-30)) + 1)[..., None]
        return np.einsum('nki,kj->nji', delta, weights) / tps['scale'] + affine
    if value is None:
        value = reg.apply(points)
    return np.stack([(reg.apply(points + [.01, 0]) - value) / .01,
                     (reg.apply(points + [0, .01]) - value) / .01], axis=2)


def _minimum_jacobian(reg, points):
    minimum = math.inf
    for start in range(0, len(points), 4096):
        block = points[start:start + 4096]
        determinants = np.linalg.det(_jacobian(reg, block))
        if not np.isfinite(determinants).all():
            return -math.inf
        minimum = min(minimum, float(determinants.min()))
    return minimum


def _centre_residuals(reg, c2, c1, h):
    """Nearest-reference residuals in reference letter heights (no index pairing)."""
    return cKDTree(_points(c1)).query(reg.apply(c2))[0] / h


def _scored(reg, c2, c1, h):
    d = _centre_residuals(reg, c2, c1, h)
    reg.inlier_fraction = float(np.mean(d < .25))
    reg.quality = _quality(reg.inlier_fraction)
    reg.diagnostics.update(median_residual_h=float(np.median(d)), p98_residual_h=float(np.quantile(d, .98)))
    return reg


def _tps_refine(initial, c2, c1, h, probes):
    """Smooth seam correction on geometric ICP matches, never index matches."""
    best, current = initial, initial
    tree = cKDTree(c1)
    rng = np.random.default_rng(0)
    matrix = np.asarray(initial.parameters['matrix'])
    for _ in range(8):
        distance, match = tree.query(current.apply(c2))
        keep = np.flatnonzero(distance < .8 * h)
        if len(keep) < 6:
            break
        if len(keep) > 400:
            keep = rng.choice(keep, 400, replace=False)
        displacement = c1[match[keep]] - _apply(matrix, c2[keep])
        # Centre inlier counts can reward a folded TPS which collapses false
        # detections onto the same target. Increase smoothness geometrically,
        # never accept that map merely because its nearest-centre score rose.
        rejected = []
        for regularization in (1., 2., 4., 8., 16., 32., 64.):
            tps = _tps_parameters(c2[keep], displacement, h, regularization)
            candidate = Registration('piecewise', {'matrix': matrix.tolist(), 'tps': tps})
            minimum = _minimum_jacobian(candidate, probes)
            if minimum > 0:
                candidate.diagnostics['orientation_check'] = {
                    'fit_version': FIT_VERSION, 'samples': len(probes), 'minimum_determinant': minimum,
                    'rejected_regularizations': rejected, 'global_bijectivity_proven': False}
                current = _scored(candidate, c2, c1, h)
                break
            rejected.append(regularization)
        else:
            break
        if current.inlier_fraction > best.inlier_fraction:
            best = current
        if best.inlier_fraction >= .98:
            break
    return best


def _icp(source, target, h, initial, *, iterations=40):
    tree = cKDTree(target)
    matrix = np.asarray(initial, dtype=float)
    for iteration in range(iterations):
        d, j = tree.query(_apply(matrix, source))
        keep = d <= np.percentile(d, 70) if iteration < iterations // 2 else d < .3 * h
        if keep.sum() < 3:
            break
        updated = _similarity(source[keep], target[j[keep]])
        converged = np.max(np.abs(updated - matrix)) < 1e-7
        matrix = updated
        if converged and iteration >= iterations // 2:
            break
    return matrix


def _global(c2, c1, h, seed=0, initial=None):
    rng = np.random.default_rng(seed)
    candidates = []
    # Three independent 600-point restarts are essential for periodic letter grids.
    for _ in range(3 if len(c2) > 600 else 1):
        sub = c2[rng.choice(len(c2), 600, replace=False)] if len(c2) > 600 else c2
        starts = []
        if initial is not None:
            starts.append(initial)
        else:
            for angle in (0., math.pi / 2, -math.pi / 2, math.pi):
                rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                starts.append(np.column_stack([rotation, c1.mean(0) - rotation @ sub.mean(0)]))
        for start in starts:
            matrix = _icp(sub, c1, h, start)
            reg = _scored(Registration('similarity', {'matrix': matrix.tolist(), 'kind': 'similarity'}), c2, c1, h)
            candidates.append(reg)
    return max(candidates, key=lambda r: (r.inlier_fraction, -r.diagnostics['median_residual_h']))


def _piecewise(global_reg, c2, c1, h, *, source_size=None):
    """Try increasingly fine overlapping seam bands; stop at the simplest pass."""
    best = global_reg
    initial = np.asarray(global_reg.parameters['matrix'])
    tree = cKDTree(c1)
    probes = _source_probes(c2, h, source_size)
    # Search either seam direction; no assumption about the scanner's stitching axis.
    for count in (2, 3, 4, 6, 8, 12):
        candidates = []
        for axis in (0, 1):
            knots = np.linspace(c2[:, axis].min(), c2[:, axis].max(), count)
            if knots[-1] - knots[0] < h:
                continue
            width = (knots[1] - knots[0]) * 1.1
            matrices = []
            for knot in knots:
                local = c2[np.abs(c2[:, axis] - knot) <= width]
                if len(local) < 6:
                    matrices.append(initial)
                    continue
                # Broader early matching than the audit is necessary at a stitched seam.
                matrix = initial.copy()
                for _ in range(30):
                    d, j = tree.query(_apply(matrix, local))
                    keep = d < max(.8 * h, np.quantile(d, .7))
                    if keep.sum() < 4:
                        break
                    updated = _similarity(local[keep], c1[j[keep]])
                    if np.max(np.abs(updated - matrix)) < 1e-6:
                        break
                    matrix = updated
                matrices.append(matrix)
            reg = Registration('piecewise', {'matrix': initial.tolist(), 'axis': axis,
                                             'knots': knots.tolist(), 'matrices': np.asarray(matrices).tolist()})
            if _minimum_jacobian(reg, probes) > 0:
                candidates.append(_scored(reg, c2, c1, h))
        if candidates:
            candidate = max(candidates, key=lambda r: (r.inlier_fraction, -r.diagnostics['median_residual_h']))
            if candidate.inlier_fraction > best.inlier_fraction:
                best = candidate
            if best.inlier_fraction >= .98:
                return best
    return _tps_refine(best, c2, c1, h, probes)


def fit_centres(c2, c1, h, *, merged=False, seed=0):
    """Fit Rotation2 -> Rotation1 from letter centres alone (letter height `h`).

    `merged` (a file-name property of stitched scans) forces local-model trials. The global fit uses four
    quarter-turn initialisations x three subsample restarts; a heavily stitched regular grid can make the wrong
    quarter-turn win globally, so a poor fit is retried with local refinement from every rotation.
    """
    c2, c1 = _points(c2), _points(c1)
    if min(len(c2), len(c1)) < 3 or not np.isfinite(h) or h <= 0:
        raise ValueError('registration requires >= 3 centres per view and positive finite h')
    reg = _global(c2, c1, h, seed)
    fraction = reg.inlier_fraction
    if merged or fraction < .98:
        reg = _piecewise(reg, c2, c1, h)
        if reg.inlier_fraction < .98:
            for angle in (0., math.pi / 2, -math.pi / 2, math.pi):
                rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                initial = np.column_stack([rotation, c1.mean(0) - rotation @ c2.mean(0)])
                alternative = _global(c2, c1, h, seed, initial)
                alternative = _piecewise(alternative, c2, c1, h)
                if alternative.inlier_fraction > reg.inlier_fraction:
                    reg = alternative
                if reg.inlier_fraction >= .98:
                    break
    reg.diagnostics['global_inlier_fraction'] = fraction
    return reg


def _refine_with_centres(reg, c2, c1, h, *, merged):
    """Trimmed ICP initialised by the image fit `reg` (similarity or `grid`), on detected letter centres."""
    c2, c1 = _points(c2), _points(c1)
    if min(len(c2), len(c1)) < 3 or h <= 0 or not np.isfinite(h):
        raise ValueError('registration requires >= 3 centres per view and positive finite h')
    # For a piecewise seed fit the residual in its already warped frame.
    if reg.model == 'piecewise':
        correction = _global(reg.apply(c2), c1, h, initial=np.eye(3)[:2])
        params = dict(reg.parameters)
        correction_matrix = _homogeneous(correction.parameters['matrix'])
        params['matrix'] = (correction_matrix @ _homogeneous(params['matrix']))[:2].tolist()
        grid = dict(params['grid'])
        matrices = np.asarray(grid['matrices'])
        transformed = [(correction_matrix @ _homogeneous(m))[:2] for m in matrices.reshape(-1, 2, 3)]
        grid['matrices'] = np.asarray(transformed).reshape(matrices.shape).tolist()
        params['grid'] = grid
        result = _scored(Registration('piecewise', params), c2, c1, h)
    else:
        result = _global(c2, c1, h, initial=reg.parameters['matrix'])
    original = _scored(Registration(reg.model, dict(reg.parameters), diagnostics=dict(reg.diagnostics)), c2, c1, h)
    if original.inlier_fraction > result.inlier_fraction:
        result = original
    fraction = result.inlier_fraction
    if merged or fraction < .98:
        result = _piecewise(result, c2, c1, h, source_size=reg.diagnostics.get('source_size'))
    result.diagnostics['global_inlier_fraction'] = fraction
    result.diagnostics['fit_version'] = FIT_VERSION
    return result


def _gradient(gray, scale):
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small.astype(np.float32) / 255, (0, 0), .7)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    # Limit scanner frame edges without discarding weak surface texture.
    ceiling = float(np.quantile(gradient, .995))
    return np.minimum(gradient, max(ceiling, 1e-6))


def _fit_images(image2, image1, *, merged):
    """Image-only phase-correlation coarse search followed by Euclidean/affine ECC.

    Inputs are uint8 grayscale images, not boxes, heights, or metadata. Rotations
    are 0/90/180/270 +/-5 degrees in .5 degree steps. Geometric inlier fraction
    remains unmeasured without centres; the quality flag is explicitly an
    image-confidence proxy.
    """
    for image in (image2, image1):
        if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 32:
            raise ValueError('images must be uint8 grayscale arrays at least 32 pixels wide/high')
    coarse_scale = .125
    a, b = _gradient(image2, coarse_scale), _gradient(image1, coarse_scale)
    side = max(*a.shape, *b.shape)
    side = cv2.getOptimalDFTSize(math.ceil(side * math.sqrt(2)))
    size = (side, side)
    target = np.zeros((side, side), np.float32)
    target[:b.shape[0], :b.shape[1]] = b
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    best = None
    for quarter in (0, 90, 180, 270):
        for offset in np.arange(-5., 5.01, .5):
            matrix = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), quarter + offset, 1.)
            matrix[:, 2] += [(b.shape[1] - a.shape[1]) / 2, (b.shape[0] - a.shape[0]) / 2]
            moving = cv2.warpAffine(a, matrix, size)
            # OpenCV may window the arrays in place. Reusing target would
            # progressively erase its edges and bias later rotation trials.
            shift, peak = cv2.phaseCorrelate(moving, target.copy(), window)
            if best is None or peak > best[0]:
                matrix[:, 2] += shift
                best = (peak, matrix)
    matrix = (np.linalg.inv(_scale_frame(coarse_scale)) @ _homogeneous(best[1])
              @ _scale_frame(coarse_scale))[:2]
    fine_scale = .25
    moving, target = _gradient(image2, fine_scale), _gradient(image1, fine_scale)
    fine_matrix = (_scale_frame(fine_scale) @ _homogeneous(matrix)
                   @ np.linalg.inv(_scale_frame(fine_scale)))[:2]
    aligned = cv2.warpAffine(moving, fine_matrix, (target.shape[1], target.shape[0]))
    valid = cv2.warpAffine(np.ones(moving.shape, np.uint8), fine_matrix, (target.shape[1], target.shape[0]))
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
    correction = np.eye(3, dtype=np.float32)[:2]
    ecc_scores, errors = [], []
    kind = 'similarity'
    for motion, name in ((cv2.MOTION_EUCLIDEAN, 'euclidean'), (cv2.MOTION_AFFINE, 'affine')):
        previous = correction.copy()
        try:
            score, correction = cv2.findTransformECC(target, aligned, correction, motion,
                                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                                                     inputMask=valid, gaussFiltSize=5)
            ecc_scores.append({'motion': name, 'correlation': float(score)})
            if name == 'affine':
                kind = 'affine'
        except cv2.error as exc:
            correction = previous
            errors.append({'motion': name, 'error': str(exc).splitlines()[-1]})
    # ECC returns template -> aligned (inverse-map convention).
    result = np.linalg.inv(_homogeneous(correction)) @ _homogeneous(fine_matrix)
    result = (np.linalg.inv(_scale_frame(fine_scale)) @ result @ _scale_frame(fine_scale))[:2]
    reg = Registration('similarity', {'matrix': result.tolist(), 'kind': kind},
                       diagnostics={'coarse_peak': float(best[0]), 'ecc': ecc_scores, 'ecc_errors': errors,
                                    'preprocess': False, 'source_size': [image2.shape[1], image2.shape[0]],
                                    'reference_size': [image1.shape[1], image1.shape[0]], 'fit_version': FIT_VERSION})
    correlation = ecc_scores[-1]['correlation'] if ecc_scores else 0.
    if merged or correlation < LOCAL_ECC_THRESHOLD:
        reg = _local_ecc(reg, moving, target, fine_scale)
    reg.quality = 'image_usable' if best[0] >= .1 and correlation >= .35 else 'image_poor'
    reg.diagnostics['quality_source'] = 'image_ecc_phase_proxy'
    return reg


def _image_score(reg, moving, target, scale):
    """Image-only sampled NCC, evaluating the actual blended transform."""
    stride = max(1, math.ceil(max(target.shape) / 512))
    yy, xx = np.mgrid[2:target.shape[0] - 2:stride, 2:target.shape[1] - 2:stride]
    target_small = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    target_pixels = _apply(np.linalg.inv(_scale_frame(scale))[:2], target_small)
    inverse = np.linalg.inv(_homogeneous(reg.parameters['matrix']))[:2]
    source = _apply(inverse, target_pixels)
    for _ in range(12 if reg.model == 'piecewise' else 0):
        residual = target_pixels - reg.apply(source)
        source += residual @ inverse[:, :2].T
        if np.max(np.abs(residual)) < .05:
            break
    source_small = _apply(_scale_frame(scale)[:2], source)
    valid = ((source_small[:, 0] >= 1) & (source_small[:, 0] < moving.shape[1] - 2)
             & (source_small[:, 1] >= 1) & (source_small[:, 1] < moving.shape[0] - 2))
    sampled = cv2.remap(moving, source_small[:, 0].astype(np.float32).reshape(yy.shape),
                        source_small[:, 1].astype(np.float32).reshape(yy.shape), cv2.INTER_LINEAR).ravel()
    a, b = sampled[valid], target[yy.ravel(), xx.ravel()][valid]
    if len(a) < 20:
        return -1., float(valid.mean())
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)), float(valid.mean())


def _local_ecc(reg, moving, target, scale):
    """Overlapping 3x3 image ECC patches, bilinearly blended, no letter geometry."""
    matrix = (_scale_frame(scale) @ _homogeneous(reg.parameters['matrix'])
              @ np.linalg.inv(_scale_frame(scale)))[:2]
    size = (target.shape[1], target.shape[0])
    aligned = cv2.warpAffine(moving, matrix, size)
    valid = cv2.warpAffine(np.ones(moving.shape, np.uint8), matrix, size)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
    x_knots = np.linspace(.15 * size[0], .85 * size[0], 3)
    y_knots = np.linspace(.15 * size[1], .85 * size[1], 3)
    matrices, diagnostics = [], []
    for y in y_knots:
        row = []
        for x in x_knots:
            x0, x1 = max(0, int(x - .3 * size[0])), min(size[0], int(x + .3 * size[0]))
            y0, y1 = max(0, int(y - .3 * size[1])), min(size[1], int(y + .3 * size[1]))
            a, b = aligned[y0:y1, x0:x1].copy(), target[y0:y1, x0:x1].copy()
            mask = valid[y0:y1, x0:x1].copy()
            correction = np.eye(3, dtype=np.float32)[:2]
            entry = {'patch': [x0, y0, x1, y1], 'ecc': []}
            if min(a.shape) >= 32 and np.mean(mask) > .5:
                shift, peak = cv2.phaseCorrelate(a.copy(), b.copy(),
                                                cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F))
                entry['phase_peak'] = float(peak)
                if peak > .1 and abs(shift[0]) < .15 * a.shape[1] and abs(shift[1]) < .15 * a.shape[0]:
                    correction[:, 2] = -np.asarray(shift)
                for motion in (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE):
                    previous = correction.copy()
                    try:
                        score, correction = cv2.findTransformECC(
                            b, a, correction, motion,
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5),
                            inputMask=mask, gaussFiltSize=5)
                        entry['ecc'].append(float(score))
                    except cv2.error:
                        correction = previous
                        entry['ecc_error'] = True
            translation = np.array([[1., 0., x0], [0., 1., y0], [0., 0., 1.]])
            local = translation @ np.linalg.inv(_homogeneous(correction)) @ np.linalg.inv(translation)
            effective = local @ _homogeneous(matrix)
            original = np.linalg.inv(_scale_frame(scale)) @ effective @ _scale_frame(scale)
            row.append(original[:2].tolist())
            diagnostics.append(entry)
        matrices.append(row)
    grid = {'frame': reg.parameters['matrix'], 'matrices': matrices}
    # Convert resize coordinates with the same pixel-centre convention as matrices.
    grid['x_knots'] = ((x_knots - (scale - 1) / 2) / scale).tolist()
    grid['y_knots'] = ((y_knots - (scale - 1) / 2) / scale).tolist()
    candidate = Registration('piecewise', {'matrix': reg.parameters['matrix'], 'grid': grid})
    global_score, global_overlap = _image_score(reg, moving, target, scale)
    local_score, local_overlap = _image_score(candidate, moving, target, scale)
    source_size = reg.diagnostics['source_size']
    probes = _source_probes(np.empty((0, 2)), max(source_size) / 25.6, source_size)
    minimum = _minimum_jacobian(candidate, probes)
    accepted = local_score > global_score and local_overlap >= .95 * global_overlap and minimum > 0
    result = candidate if accepted else reg
    result.diagnostics = dict(reg.diagnostics)
    result.diagnostics['local_ecc'] = {'accepted': accepted, 'global_ncc': global_score, 'local_ncc': local_score,
                                       'global_overlap': global_overlap, 'local_overlap': local_overlap,
                                       'minimum_sampled_jacobian': minimum if math.isfinite(minimum) else None,
                                       'patches': diagnostics, 'threshold': LOCAL_ECC_THRESHOLD}
    return result


def register_pair(gray1, gray2, centres1=None, centres2=None, *, letter_height=None, merged=False):
    """Fit Rotation2 -> Rotation1 for two uint8 grayscale views; no annotation is read.

    Without centres only image content determines the transform (quality: image-confidence proxy). With the
    letter centres detected in both views and their median letter height, the image fit is refined on the
    centres and carries a measured inlier fraction and quality. `merged` is a file-name property of stitched scans.
    """
    if (centres1 is None) != (centres2 is None):
        raise ValueError('supply centres for both views or neither')
    reg = _fit_images(gray2, gray1, merged=merged)
    if centres1 is None:
        return reg
    if letter_height is None:
        raise ValueError('centre refinement requires the detected letter height')
    result = _refine_with_centres(reg, centres2, centres1, letter_height, merged=merged)
    result.diagnostics['height_source'] = 'provided_detection_height'
    result.diagnostics['letter_height'] = float(letter_height)
    return result

"""The Ultralytics boundary of the detector: the pinned version, dense inference routing, prediction on tiles.

The detector is YOLO26-L with an oriented-box head that has two branches, one-to-many (trained with a classical
assignment, needs NMS) and one-to-one (trained NMS-free, capped at `max_det` boxes per tile). Letters are packed too
densely for a cap, so inference uses the one-to-many branch, decodes every anchor and runs Ultralytics' rotated NMS
(`iou`) at `max_det` boxes per tile; training keeps both branches.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def require(version: str) -> None:
    """Raise unless the installed Ultralytics is `version` (the recipe is defined against one release)."""
    import ultralytics

    if ultralytics.__version__ != version:
        raise RuntimeError(f'expected Ultralytics {version}, found {ultralytics.__version__}')


def configure_dense(native, max_det: int):
    """Route the native OBB model's inference through its one-to-many branch, keeping the training branches."""
    from ultralytics.nn.modules.head import OBB26
    from ultralytics.nn.tasks import OBBModel

    head = native.model[-1] if isinstance(native, OBBModel) else None
    if type(head) is not OBB26:
        raise TypeError('the detector must be a YOLO26 OBB model (OBB26 head)')
    native.end2end = False
    head.max_det = max_det
    return native


def load(weights: Path, max_det: int, device: str):
    """The checkpoint `weights` as an Ultralytics model on `device`, routed for dense inference."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    configure_dense(model.model, max_det)
    model.to(device)
    return model


def detect(model, tiles: np.ndarray, infer: dict, device: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Detections of each tile of `tiles` (n, side, side, 3 uint8 RGB): (quads (k, 4, 2) float32 in tile pixels, in
    Ultralytics' corner order; scores (k,) float32). `infer` is the 'infer' configuration section."""
    batch = torch.from_numpy(tiles).to(device).permute(0, 3, 1, 2).contiguous().float().div_(255.)
    with torch.inference_mode():
        # nms=None selects the one-to-many branch with external rotated NMS (nms=False would select the capped
        # one-to-one branch).
        results = model.predict(batch, conf=infer['conf'], iou=infer['nms_iou'], max_det=infer['max_det'],
                                imgsz=tiles.shape[1], device=device, verbose=False, nms=None)
    if model.predictor.model.end2end is not False:
        raise RuntimeError('dense inference was routed to the capped one-to-one branch')
    if len(results) != len(tiles):
        raise RuntimeError('the model did not return one result per tile')
    found = []
    for result in results:
        if result.obb is None:
            found.append((np.empty((0, 4, 2), np.float32), np.empty(0, np.float32)))
        else:
            found.append((result.obb.xyxyxyxy.detach().cpu().numpy(), result.obb.conf.detach().cpu().numpy()))
    return found

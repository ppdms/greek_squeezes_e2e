"""Letter detection: the detector's training, its inference on pages and the calibration of its scores.

Chain k's detector is YOLO26-L (oriented boxes) trained on the squeezes outside fold k; the reported detectors' folds
are those of data/folds_train181.json (the 181 TRAIN squeezes), which each stage takes as `--folds-file`. Stages, each
a module with a command line (`python -m trogs26.detect.<stage> --help`):

  train      tiles of the training squeezes -> models/detector/f{k}/{weights.pt, receipt.json}
  infer      both views of a page -> chains/f{k}/{oof,new}/det_raw/{base}.json   (uncalibrated)
  calibrate  every fold's out-of-fold raw records + annotations -> detector/calibration.json and
             chains/f{k}/{oof,new}/det/{base}.json   (the records the later stages read)
  frozen     the detector work tree's calibrated records and detector/calibration.json -> the chains of another
             work tree
  redetect   the pages whose registration `trogs26.registration_images` replaced -> detected again through it

Modules: `page` (scale frames, page rendering, tiling, box geometry), `tiles` (training tiles and targets),
`augment`/`trainer` (training), `yolo` (the Ultralytics boundary), `fusion` (the three views' detections into one set),
`records` (the record layout), `annotations` (public letter boxes).
"""

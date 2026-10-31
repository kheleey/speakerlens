# Bundled third-party code (Apache-2.0)

Only what is needed to run a pretrained WeSep target-speaker-extraction model (BSRNN) for
`speakerlens.extraction`, copied unchanged unless noted, so nothing extra has to be installed
(and pip can't replace a CUDA build of torch):

- `wesep/` from https://github.com/wenet-e2e/wesep (`wesep_deprecated/`, the version the
  released `bsrnn_ecapa_vox1` checkpoint was trained with), Apache-2.0.
- `wespeaker/models/` from https://github.com/wenet-e2e/wespeaker, Apache-2.0.
  `speaker_model.py` is slimmed to the speaker encoders BSRNN uses.

See LICENSE (Apache License 2.0). These folders are imported as top-level `wesep` and
`wespeaker` only while the model loads (see `extraction._import_vendor`).

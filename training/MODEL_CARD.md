# tse-alimeeting: speaker extraction fine-tuned on AliMeeting

A WeSep BSRNN target speaker extraction model (`bsrnn_ecapa_vox1`, trained on VoxCeleb1
mixtures) fine-tuned for 20,000 steps on the AliMeeting training meetings: the table
microphone is the input and each person's headset the target. Files: `avg_model.pt` and
`config.yaml`, loaded by speakerlens when `TSE_MODEL` points at this folder.

**Results** with the speakerlens pipeline (tcpWER, lower is better):

| | Released model | This model |
| --- | --- | --- |
| AliMeeting Eval (8 meetings) | 0.278 (DER 0.224) | 0.275 (DER 0.219) |
| AliMeeting Test (20 meetings) | 0.288 (DER 0.226) | 0.281 (DER 0.219) |
| ASCEND conversations (9) | 0.174 | unchanged |

The snapshot was chosen on Eval; Test was used only to confirm it. The model was trained on
Mandarin meetings recorded by a table microphone; on other recordings the released model may
be the safer choice.

**Training:** `training/` in https://github.com/kheleey/speakerlens (headset and table
microphone lined up per window with GCC-PHAT; voice samples made as the pipeline makes them,
30% deliberately contaminated with another voice; SI-SDR loss; speaker encoder frozen).

**License:** CC-BY-SA-4.0. Built from:
- AliMeeting: F. Yu et al., "M2MeT: The ICASSP 2022 Multi-Channel Multi-Party Meeting
  Transcription Challenge", ICASSP 2022. CC-BY-SA-4.0. https://www.openslr.org/119/
- WeSep `bsrnn_ecapa_vox1` (https://github.com/wenet-e2e/wesep), trained on VoxCeleb1,
  CC-BY-4.0.

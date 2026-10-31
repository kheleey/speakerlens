# Slimmed from wespeaker/models/speaker_model.py (Apache-2.0): only the speaker encoders
# WeSep's BSRNN checkpoints use, so no extra packages are needed.
import wespeaker.models.campplus as campplus
import wespeaker.models.ecapa_tdnn as ecapa_tdnn
import wespeaker.models.resnet as resnet


def get_speaker_model(model_name: str):
    if model_name.startswith("ECAPA_TDNN"):
        return getattr(ecapa_tdnn, model_name)
    if model_name.startswith("ResNet"):
        return getattr(resnet, model_name)
    if model_name.startswith("CAMPPlus"):
        return getattr(campplus, model_name)
    raise ValueError(f"speaker model {model_name} is not bundled")

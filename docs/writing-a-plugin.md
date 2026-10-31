# Writing a language plugin

A plugin wraps one speech-to-text model. It receives one speaker turn at a time and
returns text, optional word timings and a raw confidence score. You don't need to know
anything about diarization, routing or summaries.

## 1. Implement the contract

```python
# my_plugin/plugin.py
import numpy as np

from speakerlens.plugins import LanguagePlugin, PluginManifest, PluginResult, TurnContext
from speakerlens.types import Word


class IndonesianPlugin(LanguagePlugin):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None
        self.manifest = PluginManifest(
            id="my-indonesian",
            languages=["id"],  # ISO 639-1 codes this plugin handles
            model="org/my-indonesian-asr",
            license="Apache-2.0",  # license of the model weights
            priority=10,  # beats the Whisper fallback for "id"
            requires=["my_asr_library"],  # plugin is skipped if these aren't installed
        )

    def load(self) -> None:
        if self._model is None:
            import my_asr_library

            self._model = my_asr_library.load("org/my-indonesian-asr", device=self.device)

    def transcribe(self, audio: np.ndarray, sample_rate: int, ctx: TurnContext) -> PluginResult:
        self.load()
        out = self._model.recognize(audio, sample_rate)
        return PluginResult(
            text=out.text,
            # times are seconds relative to the start of `audio`
            words=[
                Word(text=w.text, start=w.start, end=w.end, confidence=w.prob) for w in out.words
            ],
            raw_confidence=out.score,  # any 0..1 score; calibration fixes its scale
            language="id",
        )
```

Rules:

- `audio` is 16 kHz mono float32. One call = one turn by one speaker, usually under 30 s.
- Return times relative to the audio you received; the pipeline shifts them.
- Return `raw_confidence=None` if your model has no score. The plugin still works, but it
  can't win a "top two" comparison against a plugin that has one.
- Load models lazily (in `load()` or the first `transcribe()`), so listing plugins stays fast.
- For a plugin that handles code-switching between its languages inside one turn, set
  `mixed=True` and list all of them, e.g. `languages=["ms", "en"]`.

## 2. Register it

In your package's `pyproject.toml`:

```toml
[project.entry-points."speakerlens.plugins"]
indonesian = "my_plugin.plugin:create_plugins"
```

```python
def create_plugins(settings):
    from speakerlens.config import resolve_device

    return [IndonesianPlugin(device=resolve_device(settings.device))]
```

After `pip install -e .` in your plugin package, `speakerlens plugins` lists it.

## 3. Calibrate it

Make a JSONL file of labeled clips (aim for 30+ minutes, 200+ clips):

```json
{"audio": "clips/001.wav", "text": "reference transcript"}
{"audio": "long.wav", "start": 12.5, "end": 18.0, "text": "another reference"}
```

```bash
speakerlens calibrate my-indonesian calibration.jsonl
```

This fits an isotonic mapping from your raw scores to "probability the turn is right"
(error rate at most 10%) and prints the expected calibration error before and after.
The pipeline applies it automatically from then on.

## 4. Benchmark it

Measure error rates on a test set in your language before and after your plugin replaces
the fallback (`bench/run_bench.py` scores any folder of `audio/` + `ref/` files). A pull
request with numbers is much easier to accept.

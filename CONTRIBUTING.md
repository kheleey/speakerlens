# Contributing

Thanks for helping! The easiest ways to contribute:

1. **Add a language plugin**: see [docs/writing-a-plugin.md](docs/writing-a-plugin.md).
2. **Improve a plugin**, e.g. confidence scores or word timestamps for SenseVoice.
3. **Report a bad transcript**: open an issue with the language(s), audio conditions and
   what went wrong (no private recordings, please).

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # tests don't need the model libraries
pytest
ruff check . && ruff format .
mypy src
```

Pull requests should keep `pytest`, `ruff check` and `mypy src` passing. Changes that affect
accuracy should include numbers before and after (`bench/` has the scripts behind the
README's AliMeeting and ASCEND numbers).

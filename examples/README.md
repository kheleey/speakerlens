# Examples

Put test recordings here (they're git-ignored, so they won't be committed). For example:

```
examples/
  sample.m4a           # a short mixed Mandarin-English conversation
  voices/
    Alice.m4a          # 10-30 seconds of Alice speaking alone
    Bob.m4a
```

Then:

```bash
speakerlens transcribe examples/sample.m4a --voices examples/voices --translate-to en
```

Only record people who agreed to it.

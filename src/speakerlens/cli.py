"""Command-line interface: `speakerlens --help`."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from speakerlens import __version__
from speakerlens.config import Settings, load_settings

app = typer.Typer(
    help="Speaker-aware, multi-language transcription with translation and per-person summaries.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

EnvFile = Annotated[
    Path | None, typer.Option("--env-file", help="Path to a .env file (default: search from cwd).")
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def _settings(env_file: Path | None) -> Settings:
    try:
        return load_settings(env_file)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _profile_label(settings: Settings) -> str:
    from speakerlens.config import resolve_device
    from speakerlens.plugins.registry import resolve_profile

    resolved = resolve_profile(settings.asr_profile, resolve_device(settings.device))
    return resolved if resolved == settings.asr_profile else f"{settings.asr_profile} -> {resolved}"


def _preflight(settings: Settings, languages: list[str]) -> None:
    """Fail early with a clear message instead of a stack trace mid-run."""
    from speakerlens.plugins.registry import default_registry

    problems = []
    if not shutil.which("ffmpeg"):
        problems.append("ffmpeg is not installed: sudo apt install ffmpeg")
    if importlib.util.find_spec("pyannote") is None:
        problems.append(escape("Diarization is not installed: pip install -e '.[diarization]'"))
    elif not settings.hf_token:
        problems.append("HF_TOKEN is empty in .env (needed to download the pyannote model)")
    if not default_registry(settings).all():
        problems.append(escape("No language plugins installed: pip install -e '.[all]'"))
    needs_langid = len(languages) != 1 or languages == ["auto"]
    if needs_langid and importlib.util.find_spec("faster_whisper") is None:
        problems.append(
            escape("Detecting languages needs Whisper: pip install -e '.[whisper]' ")
            + "(or pass a single --lang)"
        )
    if problems:
        console.print("[red]Can't start yet:[/red]")
        for p in problems:
            console.print(f"  - {p}")
        console.print("Run [bold]speakerlens doctor[/bold] for the full check.")
        raise typer.Exit(1)


@app.command()
def transcribe(
    audio: Annotated[
        Path, typer.Argument(help="Audio or video file.", exists=True, dir_okay=False)
    ],
    lang: Annotated[
        list[str],
        typer.Option("--lang", "-l", help="Spoken language(s), repeatable. Use 'auto' for any."),
    ] = ["zh", "en"],  # noqa: B006
    translate_to: Annotated[
        str | None, typer.Option("--translate-to", "-t", help="Translate every turn, e.g. en.")
    ] = None,
    summary: Annotated[bool, typer.Option(help="Write per-speaker summaries.")] = True,
    summary_lang: Annotated[
        str | None, typer.Option(help="Summary language (default: --translate-to, else en).")
    ] = None,
    voices: Annotated[
        Path | None,
        typer.Option(help="Folder of voice samples named after each person, e.g. voices/Alice.wav"),
    ] = None,
    speakers: Annotated[
        int | None, typer.Option(help="Exact number of speakers, if known.")
    ] = None,
    min_speakers: Annotated[int | None, typer.Option()] = None,
    max_speakers: Annotated[int | None, typer.Option()] = None,
    extract: Annotated[
        bool | None,
        typer.Option(
            "--extract/--no-extract",
            help="Recognize people talking over each other from each person's extracted voice "
            "(target speaker extraction). Default: SPEAKER_EXTRACTION, else on.",
        ),
    ] = None,
    reassign_margin: Annotated[
        float | None,
        typer.Option(
            help="Speaker check: move turns that clearly match another speaker's voice "
            "(default: SPEAKER_REASSIGN_MARGIN, else 0.1; 0 = off)."
        ),
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output folder.")] = Path("output"),
    formats: Annotated[
        list[str], typer.Option("--format", "-f", help="md, json, srt, vtt, rttm, stm")
    ] = ["md", "json", "srt"],  # noqa: B006
    cache: Annotated[bool, typer.Option(help="Reuse cached diarization.")] = True,
    asr_profile: Annotated[
        str | None,
        typer.Option(help="Speech models: auto, routing or nano (default: ASR_PROFILE)."),
    ] = None,
    env_file: EnvFile = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Transcribe a recording: who said what, in which language, plus summaries."""
    _setup_logging(verbose)
    if asr_profile:
        os.environ["ASR_PROFILE"] = asr_profile
    if extract is not None:
        os.environ["SPEAKER_EXTRACTION"] = "on" if extract else "off"
    from speakerlens.export import write_outputs
    from speakerlens.pipeline import Pipeline, PipelineOptions

    settings = _settings(env_file)
    options = PipelineOptions(
        languages=[x.lower() for x in lang],
        translate_to=translate_to.lower() if translate_to else None,
        summarize=summary,
        summary_language=summary_lang,
        voices_dir=voices,
        num_speakers=speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        reassign_margin=(settings.reassign_margin if reassign_margin is None else reassign_margin)
        or None,
        use_cache=cache,
    )
    _preflight(settings, options.languages)
    if settings.llm.enabled and not settings.llm.runs_locally and (translate_to or summary):
        console.print(
            f"[yellow]Note:[/yellow] transcript text will be sent to {settings.llm.provider} "
            "for translation/summaries. Use LLM_PROVIDER=ollama to stay fully local."
        )

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        tasks: dict[str, TaskID] = {}

        def on_progress(stage: str, done: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = progress.add_task(stage, total=max(total, 1))
            progress.update(tasks[stage], completed=done, total=max(total, 1))

        pipeline = Pipeline.from_settings(settings, options, progress=on_progress)
        transcript = pipeline.run(audio, options)

    paths = write_outputs(transcript, out, audio.stem, formats)
    console.print(
        f"[green]Done.[/green] {len(transcript.turns)} turns, {len(transcript.speakers)} speakers."
    )
    for p in paths:
        console.print(f"  {p}")


@app.command()
def plugins(env_file: EnvFile = None) -> None:
    """List language plugins and which languages they cover."""
    from speakerlens.plugins.registry import default_registry

    registry = default_registry(_settings(env_file))
    table = Table(title=f"Language plugins (ASR profile: {_profile_label(_settings(env_file))})")
    for col in ("id", "languages", "type", "model", "license"):
        table.add_column(col)
    for p in registry.all():
        m = p.manifest
        kind = "mixed" if m.mixed else "fallback" if m.fallback else "specialist"
        table.add_row(m.id, ", ".join(m.languages), kind, m.model, m.license)
    console.print(table)
    for pid, reason in registry.unavailable.items():
        console.print(f"[yellow]Not available:[/yellow] {pid} ({reason})")


@app.command()
def licenses(env_file: EnvFile = None) -> None:
    """Show the license of every model the installed plugins use."""
    from speakerlens.plugins.registry import default_registry

    settings = _settings(env_file)
    registry = default_registry(settings)
    console.print(f"Diarization: {settings.diarization_model} (CC-BY-4.0, attribution required)")
    console.print(f"Speaker embeddings: {settings.embedding_model} (CC-BY-4.0)")
    console.print(
        "Speaker extraction: WeSep BSRNN bsrnn_ecapa_vox1 (code Apache-2.0; checkpoint trained "
        "on VoxCeleb1, CC-BY-4.0, attribution required)"
    )
    for p in registry.all():
        console.print(f"{p.id}: {p.manifest.model} ({p.manifest.license})")


@app.command()
def doctor(env_file: EnvFile = None) -> None:
    """Check that ffmpeg, models, GPU and API keys are set up."""
    settings = _settings(env_file)
    table = Table(title=f"speakerlens {__version__} setup check")
    table.add_column("check")
    table.add_column("status")
    ok, warn, bad = "[green]ok[/green]", "[yellow]missing[/yellow]", "[red]missing[/red]"

    table.add_row("Python", platform.python_version())
    table.add_row("ASR profile", _profile_label(settings))
    table.add_row("ffmpeg", ok if shutil.which("ffmpeg") else bad + " (sudo apt install ffmpeg)")
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "no CUDA GPU, using CPU"
        table.add_row("torch", f"{torch.__version__} · {gpu}")
    except ImportError:
        table.add_row("torch", warn)
    except Exception as exc:  # e.g. broken CUDA libraries
        table.add_row("torch", f"[red]error[/red] {escape(str(exc)[:80])}")
    for module, extra in (
        ("pyannote.audio", "diarization"),
        ("faster_whisper", "whisper"),
        ("funasr", "sensevoice"),
    ):
        found = importlib.util.find_spec(module.split(".")[0]) is not None
        hint = escape(f"(pip install -e '.[{extra}]')")
        table.add_row(module, ok if found else f"{warn} {hint}")
    table.add_row(
        "HF_TOKEN",
        ok if settings.hf_token else f"{bad} (needed to download pyannote; see .env.example)",
    )
    llm = settings.llm
    if llm.enabled:
        key = (llm.api_key or "")[-4:]
        detail = f"{llm.provider} · {llm.model}" + (
            f" · key ...{key}" if llm.provider == "deepseek" else ""
        )
        table.add_row("LLM (translation, summaries)", detail)
    else:
        table.add_row(
            "LLM (translation, summaries)",
            f"{warn} (set DEEPSEEK_API_KEY in .env, or LLM_PROVIDER=ollama)",
        )
    console.print(table)


@app.command()
def calibrate(
    plugin_id: Annotated[str, typer.Argument(help="Plugin to calibrate, e.g. whisper-en.")],
    manifest: Annotated[
        Path,
        typer.Argument(
            help='JSONL file, one line per clip: {"audio": "...", "start": 0, "end": 5.2, '
            '"text": "reference transcript"}',
            exists=True,
        ),
    ],
    max_error: Annotated[
        float, typer.Option(help="A clip counts as correct when its error rate is at most this.")
    ] = 0.1,
    env_file: EnvFile = None,
) -> None:
    """Fit a confidence calibrator for one plugin from labeled clips."""
    from speakerlens.audio import SAMPLE_RATE, load_audio, slice_audio
    from speakerlens.calibration import Calibrator
    from speakerlens.metrics import error_rate
    from speakerlens.plugins.base import TurnContext
    from speakerlens.plugins.registry import default_registry

    settings = _settings(env_file)
    plugin = default_registry(settings).get(plugin_id)
    m = plugin.manifest
    language = None if (m.mixed or m.fallback) else m.languages[0]

    loaded: dict[str, np.ndarray] = {}
    raw: list[float] = []
    correct: list[bool] = []
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    with Progress(console=console) as progress:
        task = progress.add_task("Running plugin", total=len(lines))
        for i, line in enumerate(lines):
            item = json.loads(line)
            audio_path = str(Path(item["audio"]).expanduser())
            if audio_path not in loaded:
                loaded[audio_path] = load_audio(audio_path)
            audio = loaded[audio_path]
            start = float(item.get("start", 0.0))
            end = float(item["end"]) if item.get("end") is not None else len(audio) / SAMPLE_RATE
            clip = slice_audio(audio, start, end)
            ctx = TurnContext(f"C{i:05d}", "calibration", start, end, language)
            res = plugin.transcribe(clip, SAMPLE_RATE, ctx)
            if res.raw_confidence is not None:
                raw.append(res.raw_confidence)
                correct.append(error_rate(item["text"], res.text) <= max_error)
            progress.advance(task)

    if not raw:
        console.print(f"[red]{plugin_id} reports no confidence scores; nothing to calibrate.[/red]")
        raise typer.Exit(1)
    calibrator = Calibrator.fit(plugin_id, raw, correct)
    saved = calibrator.save(settings.calibration_dir)
    console.print(
        f"Saved {saved}\n{len(raw)} clips, {sum(correct)} correct. "
        f"ECE before: {calibrator.ece_before:.3f}, after: {calibrator.ece_after:.3f}"
    )


if __name__ == "__main__":
    app()

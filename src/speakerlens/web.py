"""Local web page: upload a recording, follow its progress, read who said what.

`speakerlens web` serves it on http://127.0.0.1:7860. Recordings and results stay on this
machine, in the output folder; the only thing sent anywhere is transcript text to the LLM
when DeepSeek is configured, as with `speakerlens transcribe`. Recordings are processed one
at a time, so a single GPU is never shared between two runs.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import mimetypes
import queue
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders, UploadFile
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from speakerlens import __version__
from speakerlens.config import Settings
from speakerlens.export import write_outputs
from speakerlens.pipeline import Pipeline, PipelineOptions, ProgressFn
from speakerlens.types import Transcript

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("web_page.html")
FORMATS = ["md", "json", "srt", "vtt"]
LANGUAGES = {"zh+en": ["zh", "en"], "zh": ["zh"], "en": ["en"], "auto": ["auto"]}
MAX_SPEAKERS = 20
JOB_FILE = "job.json"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
MEDIA_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr", ".aif", ".aiff",
    ".caf", ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".3gp",
}  # fmt: skip
MAX_UPLOAD_BYTES = 4 * 1024**3
SECURITY_HEADERS = {
    "content-security-policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    ),
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
}


class Runner(Protocol):
    progress: ProgressFn

    def run(self, source: str | Path, options: PipelineOptions) -> Transcript: ...


Factory = Callable[[Settings, PipelineOptions], Runner]


class PipelineCache:
    """Builds the real pipeline and keeps the last one, so its models stay loaded between
    recordings that need the same setup. Only one is kept, to leave GPU memory free."""

    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._pipeline: Pipeline | None = None

    def __call__(self, settings: Settings, options: PipelineOptions) -> Runner:
        allowed = options.allowed_languages
        key = (
            settings.extraction,
            settings.asr_profile,
            tuple(allowed) if len(allowed) == 1 else "detect",
            bool(options.translate_to) or options.summarize,
            options.voices_dir is not None or options.reassign_margin is not None,
        )
        if key != self._key or self._pipeline is None:
            self._pipeline = None
            self._pipeline = Pipeline.from_settings(settings, options)
            self._key = key
        return self._pipeline


def safe_name(name: str, default: str = "recording") -> str:
    """A file name that is safe to write: no folders, no odd characters, not hidden."""
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ]", "_", name).strip(" .")
    return name[:100] or default


def is_media(name: str) -> bool:
    return Path(name).suffix.lower() in MEDIA_EXTENSIONS


def _hostname(host: str) -> str:
    return (urlsplit(f"//{host}").hostname or "").lower()


class RequestGuard:
    """Lets only this page use the server, and adds the browser's safety headers.

    - Host check (when listening only on this computer): a website can't reach the server
      through a domain name that points at 127.0.0.1 (DNS rebinding).
    - Origin check: another website open in the same browser can't upload, rename or rerun.
    - Uploads over MAX_UPLOAD_BYTES are refused before they're read.
    """

    def __init__(self, app: ASGIApp, local_only: bool = True) -> None:
        self.app = app
        self.local_only = local_only

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = headers.get("host", "")
        origin = headers.get("origin")
        problem = None
        if self.local_only and _hostname(host) not in LOCAL_HOSTS:
            problem = (403, "This page only answers on 127.0.0.1 or localhost.")
        elif scope["method"] not in ("GET", "HEAD") and origin is not None:
            if urlsplit(origin).netloc.lower() != host.lower():
                problem = (403, "Requests from other websites are refused.")
        elif int(headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
            problem = (413, "The upload is too large.")
        if problem:
            await JSONResponse({"error": problem[1]}, status_code=problem[0])(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    response_headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


def parse_options(form: Mapping[str, str], settings: Settings) -> tuple[PipelineOptions, bool]:
    """Pipeline options and the extraction switch from the upload form."""
    languages = LANGUAGES.get(form.get("languages", "zh+en"))
    if languages is None:
        raise ValueError("Unknown language choice.")
    speakers = None
    if form.get("speakers", "").strip():
        try:
            speakers = int(form["speakers"])
        except ValueError:
            raise ValueError("Number of people must be a whole number.") from None
        if not 1 <= speakers <= MAX_SPEAKERS:
            raise ValueError(f"Number of people must be between 1 and {MAX_SPEAKERS}.")
    translate_to = form.get("translate_to", "").strip().lower() or None
    if translate_to and not re.fullmatch(r"[a-z]{2,3}(-[a-z]{2,4})?", translate_to):
        raise ValueError("Translation language must be a code like en, zh or id.")
    checked = ("on", "true", "1", "yes")
    options = PipelineOptions(
        languages=list(languages),
        translate_to=translate_to,
        summarize=form.get("summary", "").lower() in checked,
        num_speakers=speakers,
        reassign_margin=settings.reassign_margin or None,
        use_cache=True,
    )
    return options, form.get("extract", "").lower() in checked


def setup_problems(settings: Settings) -> list[str]:
    """What would stop a recording from being processed, in plain words."""
    problems = []
    if not shutil.which("ffmpeg"):
        problems.append("ffmpeg is not installed (sudo apt install ffmpeg).")
    if importlib.util.find_spec("pyannote") is None:
        problems.append("Diarization is not installed: pip install -e '.[diarization]'.")
    elif not settings.hf_token:
        problems.append("HF_TOKEN is empty in .env; it's needed to download the pyannote model.")
    try:
        from speakerlens.plugins.registry import default_registry

        if not default_registry(settings).all():
            problems.append("No speech models are installed: pip install -e '.[all]'.")
    except Exception as exc:  # a broken install shouldn't stop the page from loading
        problems.append(f"Speech models could not be checked: {exc}")
    return problems


@dataclass
class Stage:
    name: str
    done: int = 0
    total: int = 1


@dataclass
class Job:
    id: str
    name: str  # the uploaded file's name
    folder: Path
    audio: Path
    options: PipelineOptions
    extract: bool = True
    state: str = "queued"  # queued, running, done, error
    stages: list[Stage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    transcript: Transcript | None = None
    files: list[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.name).stem or "recording"

    def on_progress(self, stage: str, done: int, total: int) -> None:
        if not self.stages or self.stages[-1].name != stage:
            self.stages.append(Stage(stage))
        current = self.stages[-1]
        current.done, current.total = done, max(total, 1)

    def status(self, queue_position: int = 0) -> dict[str, Any]:
        now = self.finished or time.time()
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "queue_position": queue_position,
            "stages": [dataclasses.asdict(s) for s in self.stages],
            "warnings": self.warnings,
            "error": self.error,
            "created": self.created,
            "elapsed": round(now - self.started, 1) if self.started else 0.0,
            "files": self.files,
            "speakers_given": self.options.num_speakers,
            "can_rerun": self.audio.is_file(),
        }


class _JobLogHandler(logging.Handler):
    """Shows the run's warnings (a failed translation, extraction falling back) on the page."""

    def __init__(self, job: Job) -> None:
        super().__init__(level=logging.WARNING)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().strip()
        if message and message not in self.job.warnings:
            self.job.warnings.append(message)


class JobManager:
    """Runs uploaded recordings one after another on a background thread."""

    def __init__(
        self,
        settings: Settings,
        out_dir: str | Path,
        factory: Factory | None = None,
        start_worker: bool = True,
    ) -> None:
        self.settings = settings
        self.out_dir = Path(out_dir)
        self.factory: Factory = factory or PipelineCache()
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self.load_finished()
        if start_worker:
            threading.Thread(target=self._work, name="speakerlens-web", daemon=True).start()

    def new_job(self, name: str, options: PipelineOptions, extract: bool) -> Job:
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        folder = self.out_dir / job_id
        folder.mkdir(parents=True)
        name = safe_name(name)
        return Job(job_id, name, folder, folder / name, options, extract)

    def submit(self, job: Job) -> None:
        with self._lock:
            self.jobs[job.id] = job
        self._queue.put(job)

    def queue_position(self, job: Job) -> int:
        """How many recordings wait ahead of this one (0 when it's running or finished)."""
        if job.state != "queued":
            return 0
        with self._lock:
            return sum(
                1
                for j in self.jobs.values()
                if j.state == "running" or (j.state == "queued" and j.created < job.created)
            )

    def recent(self) -> list[Job]:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def _work(self) -> None:
        while True:
            self.run_job(self._queue.get())

    def run_job(self, job: Job) -> None:
        job.state, job.started = "running", time.time()
        settings = dataclasses.replace(self.settings, extraction="on" if job.extract else "off")
        handler = _JobLogHandler(job)
        logger = logging.getLogger("speakerlens")
        logger.addHandler(handler)
        failure: Exception | None = None
        try:
            job.on_progress("Loading models", 0, 1)
            runner = self.factory(settings, job.options)
            job.on_progress("Loading models", 1, 1)
            runner.progress = job.on_progress
            transcript = runner.run(job.audio, job.options)
            transcript.source = job.name
            job.transcript = transcript
            self.write(job)
        except Exception as exc:
            failure = exc
        finally:
            logger.removeHandler(handler)
        job.finished = time.time()
        if failure is None:
            job.state = "done"
            self._save_record(job)
        else:
            log.error("Processing %s failed", job.name, exc_info=failure)
            job.error = f"{type(failure).__name__}: {failure}"
            job.state = "error"

    def rerun(self, job: Job, speakers: int | None) -> Job:
        """The same recording and settings again, with a different number of people."""
        if speakers is not None and not 1 <= speakers <= MAX_SPEAKERS:
            raise ValueError(f"Number of people must be between 1 and {MAX_SPEAKERS}.")
        if not job.audio.is_file():
            raise ValueError("The original recording is no longer in the output folder.")
        options = dataclasses.replace(job.options, num_speakers=speakers, voices_dir=None)
        new = self.new_job(job.name, options, job.extract)
        shutil.copyfile(job.audio, new.audio)
        if job.options.voices_dir is not None and job.options.voices_dir.is_dir():
            new.options.voices_dir = new.folder / "voices"
            shutil.copytree(job.options.voices_dir, new.options.voices_dir)
        self.submit(new)
        return new

    def write(self, job: Job) -> None:
        """Write (or rewrite, after renaming speakers) the result files."""
        assert job.transcript is not None
        paths = write_outputs(job.transcript, job.folder, job.stem, FORMATS)
        job.files = [p.name for p in paths]

    def rename(self, job: Job, names: Mapping[str, str]) -> None:
        assert job.transcript is not None
        known = {s.id for s in job.transcript.speakers}
        unknown = set(names) - known
        if unknown:
            raise ValueError(f"Unknown speaker: {', '.join(sorted(unknown))}")
        for speaker in job.transcript.speakers:
            if speaker.id in names:
                speaker.name = names[speaker.id].strip()[:60] or None
        self.write(job)
        self._save_record(job)

    def _save_record(self, job: Job) -> None:
        record = {
            "id": job.id,
            "name": job.name,
            "audio": job.audio.name,
            "created": job.created,
            "started": job.started,
            "finished": job.finished,
            "warnings": job.warnings,
            "files": job.files,
            "options": {
                "languages": job.options.languages,
                "translate_to": job.options.translate_to,
                "summarize": job.options.summarize,
                "speakers": job.options.num_speakers,
                "extract": job.extract,
            },
        }
        (job.folder / JOB_FILE).write_text(json.dumps(record, ensure_ascii=False, indent=2))

    def load_finished(self) -> None:
        """Bring back recordings finished in earlier sessions, so they stay in the list."""
        if not self.out_dir.is_dir():
            return
        for record_path in sorted(self.out_dir.glob(f"*/{JOB_FILE}")):
            try:
                record = json.loads(record_path.read_text())
                folder = record_path.parent
                transcript = Transcript.model_validate_json(
                    (folder / f"{Path(Path(record['name']).name).stem}.json").read_text()
                )
                saved = record.get("options", {})
                voices = folder / "voices"
                options = PipelineOptions(
                    languages=saved.get("languages", transcript.languages),
                    translate_to=saved.get("translate_to", transcript.target_language),
                    summarize=saved.get("summarize", True),
                    num_speakers=saved.get("speakers"),
                    voices_dir=voices if voices.is_dir() else None,
                    reassign_margin=self.settings.reassign_margin or None,
                )
                job = Job(
                    record["id"],
                    Path(record["name"]).name,
                    folder,
                    folder / Path(record["audio"]).name,
                    options,
                    extract=saved.get("extract", self.settings.extraction != "off"),
                    state="done",
                    warnings=record.get("warnings", []),
                    created=record["created"],
                    started=record.get("started"),
                    finished=record.get("finished"),
                    files=record.get("files", []),
                    transcript=transcript,
                )
            except Exception as exc:
                log.warning("Skipping %s: %s", record_path.parent.name, exc)
                continue
            self.jobs[job.id] = job


def _form_strings(form: Any) -> dict[str, str]:
    return {k: v for k, v in form.multi_items() if isinstance(v, str)}


def _save_upload(upload: UploadFile, path: Path) -> None:
    with path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def create_app(manager: JobManager, local_only: bool = True) -> ASGIApp:
    settings = manager.settings

    def job_or_404(request: Request) -> Job | None:
        return manager.jobs.get(request.path_params["job_id"])

    def not_found() -> JSONResponse:
        return JSONResponse({"error": "No such recording."}, status_code=404)

    async def page(request: Request) -> Response:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    async def info(request: Request) -> Response:
        llm = settings.llm
        return JSONResponse(
            {
                "version": __version__,
                "llm": {
                    "enabled": llm.enabled,
                    "provider": llm.provider,
                    "local": llm.runs_locally,
                },
                "extract_default": settings.extraction != "off",
                "problems": await run_in_threadpool(setup_problems, settings),
            }
        )

    async def list_jobs(request: Request) -> Response:
        return JSONResponse([j.status(manager.queue_position(j)) for j in manager.recent()])

    async def create_job(request: Request) -> Response:
        form = await request.form(max_files=50, max_fields=50)
        upload = form.get("audio")
        if not isinstance(upload, UploadFile) or not upload.filename:
            return JSONResponse({"error": "Choose a recording to upload."}, status_code=400)
        voices = [v for v in form.getlist("voices") if isinstance(v, UploadFile) and v.filename]
        for f in [upload, *voices]:
            if not is_media(f.filename or ""):
                return JSONResponse(
                    {"error": f"{f.filename} isn't an audio or video file this page accepts."},
                    status_code=400,
                )
        try:
            options, extract = parse_options(_form_strings(form), settings)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        job = manager.new_job(upload.filename, options, extract)
        await run_in_threadpool(_save_upload, upload, job.audio)
        if voices:
            voices_dir = job.folder / "voices"
            voices_dir.mkdir()
            for voice in voices:
                await run_in_threadpool(
                    _save_upload, voice, voices_dir / safe_name(voice.filename or "", "voice")
                )
            job.options.voices_dir = voices_dir
        manager.submit(job)
        return JSONResponse(job.status(manager.queue_position(job)), status_code=201)

    async def job_status(request: Request) -> Response:
        job = job_or_404(request)
        return JSONResponse(job.status(manager.queue_position(job))) if job else not_found()

    async def job_result(request: Request) -> Response:
        job = job_or_404(request)
        if job is None or job.transcript is None:
            return not_found()
        return JSONResponse(job.transcript.model_dump(mode="json"))

    async def job_audio(request: Request) -> Response:
        job = job_or_404(request)
        if job is None or not job.audio.is_file():
            return not_found()
        media_type = mimetypes.guess_type(job.audio.name)[0] or ""
        if not media_type.startswith(("audio/", "video/")):
            media_type = "application/octet-stream"
        return FileResponse(job.audio, media_type=media_type)

    async def job_file(request: Request) -> Response:
        job = job_or_404(request)
        name = request.path_params["name"]
        if job is None or name not in job.files:
            return not_found()
        return FileResponse(job.folder / name, filename=name)

    async def rename_speakers(request: Request) -> Response:
        job = job_or_404(request)
        if job is None or job.transcript is None:
            return not_found()
        try:
            names = await request.json()
            if not isinstance(names, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in names.items()
            ):
                raise ValueError("Send a JSON object of speaker id to name.")
            await run_in_threadpool(manager.rename, job, names)
        except (ValueError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(job.transcript.model_dump(mode="json"))

    async def rerun_job(request: Request) -> Response:
        job = job_or_404(request)
        if job is None:
            return not_found()
        try:
            body = await request.json()
            speakers = body.get("speakers") if isinstance(body, dict) else None
            if speakers is not None and (
                isinstance(speakers, bool) or not isinstance(speakers, int)
            ):
                raise ValueError("Number of people must be a whole number.")
            new = await run_in_threadpool(manager.rerun, job, speakers)
        except (ValueError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(new.status(manager.queue_position(new)), status_code=201)

    app = Starlette(
        routes=[
            Route("/", page),
            Route("/api/info", info),
            Route("/api/jobs", list_jobs, methods=["GET"]),
            Route("/api/jobs", create_job, methods=["POST"]),
            Route("/api/jobs/{job_id}", job_status),
            Route("/api/jobs/{job_id}/result", job_result),
            Route("/api/jobs/{job_id}/audio", job_audio),
            Route("/api/jobs/{job_id}/files/{name}", job_file),
            Route("/api/jobs/{job_id}/speakers", rename_speakers, methods=["POST"]),
            Route("/api/jobs/{job_id}/rerun", rerun_job, methods=["POST"]),
        ]
    )
    return RequestGuard(app, local_only)


def serve(
    settings: Settings, out_dir: str | Path, host: str = "127.0.0.1", port: int = 7860
) -> None:
    import uvicorn

    local_only = _hostname(host) in LOCAL_HOSTS
    app = create_app(JobManager(settings, out_dir), local_only=local_only)
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)

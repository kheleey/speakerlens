"""The local web page: upload, progress, results, renaming, restarts, bad input."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import pytest

pytest.importorskip("starlette")
pytest.importorskip("multipart")

from speakerlens import web  # noqa: E402
from speakerlens.config import load_settings  # noqa: E402
from speakerlens.pipeline import Pipeline, PipelineOptions  # noqa: E402

from .conftest import FakeDiarizer, FakeLanguageID, FakeLLM  # noqa: E402

BOUNDARY = "speakerlens-test-boundary"


def summary_llm() -> FakeLLM:
    return FakeLLM(
        [
            {"translations": [{"id": "T0002", "text": "hello (translated)"}]},
            {
                "overview": "A short chat.",
                "speakers": [
                    {
                        "speaker": "SPEAKER_00",
                        "main_points": [{"text": "Greets", "turn_ids": ["T0001"]}],
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "main_points": [{"text": "Replies", "turn_ids": ["T0002"]}],
                    },
                ],
            },
        ]
    )


class FakeRunner:
    """The real pipeline with fake models, fed preloaded audio instead of decoding the upload."""

    def __init__(self, pipeline: Pipeline, audio) -> None:  # type: ignore[no-untyped-def]
        self.pipeline = pipeline
        self.audio = audio
        self.progress = pipeline.progress

    def run(self, source, options):  # type: ignore[no-untyped-def]
        self.pipeline.progress = self.progress
        assert Path(source).is_file()
        return self.pipeline.run(source, options, audio=self.audio)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("SPEAKERLENS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SPEAKER_EXTRACTION", raising=False)
    return load_settings(tmp_path / "missing.env")


@pytest.fixture
def factory(registry, silence):
    calls = []

    def make(settings, options):  # type: ignore[no-untyped-def]
        calls.append((settings, options))
        lid = FakeLanguageID([{"en": 0.99, "zh": 0.01}, {"zh": 0.99, "en": 0.01}])
        pipeline = Pipeline(
            registry=registry, diarizer=FakeDiarizer(), language_id=lid, llm=summary_llm()
        )
        return FakeRunner(pipeline, silence)

    make.calls = calls  # type: ignore[attr-defined]
    return make


def call(app, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    """One request through the ASGI app, without a network or an HTTP client library."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    messages: list[dict] = []
    sent = False

    async def receive():  # type: ignore[no-untyped-def]
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)  # the client stays connected until the response is done

    async def send(message):  # type: ignore[no-untyped-def]
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], dict((k.decode(), v.decode()) for k, v in start["headers"]), data


def multipart(fields: dict[str, str], files: list[tuple[str, str, bytes]]):  # type: ignore[no-untyped-def]
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    for name, filename, content in files:
        parts.append(
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
            + content
            + b"\r\n"
        )
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts), {"content-type": f"multipart/form-data; boundary={BOUNDARY}"}


def upload(app, fields=None, files=None):  # type: ignore[no-untyped-def]
    form = {
        "speakers": "2",
        "languages": "zh+en",
        "translate_to": "en",
        "summary": "on",
        "extract": "on",
    }
    form.update(fields or {})
    body, headers = multipart(
        form, files if files is not None else [("audio", "meeting.wav", b"RIFF-fake")]
    )
    return call(app, "POST", "/api/jobs", body, headers)


def wait(manager: web.JobManager, job_id: str) -> web.Job:
    deadline = time.time() + 20
    while time.time() < deadline:
        job = manager.jobs[job_id]
        if job.state in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError("the job did not finish")


def test_upload_runs_the_pipeline_and_serves_the_results(settings, factory, tmp_path):
    manager = web.JobManager(settings, tmp_path / "out", factory=factory)
    app = web.create_app(manager)

    status, _, data = upload(
        app, files=[("audio", "meeting.wav", b"RIFF-fake"), ("voices", "Alice.wav", b"v")]
    )
    assert status == 201
    job = wait(manager, json.loads(data)["id"])
    assert job.state == "done", job.error

    used_settings, options = factory.calls[0]
    assert used_settings.extraction == "on"
    assert options.num_speakers == 2 and options.translate_to == "en" and options.summarize
    assert options.languages == ["zh", "en"]
    assert (
        options.voices_dir is not None and (options.voices_dir / "Alice.wav").read_bytes() == b"v"
    )
    stages = [s.name for s in job.stages]
    assert {"Diarizing", "Transcribing", "Translating", "Summarizing"} <= set(stages)
    assert all(s.done == s.total for s in job.stages)

    status, _, data = call(app, "GET", f"/api/jobs/{job.id}")
    assert status == 200
    assert sorted(json.loads(data)["files"]) == [
        "meeting.en.srt",
        "meeting.json",
        "meeting.md",
        "meeting.srt",
        "meeting.vtt",
    ]

    status, _, data = call(app, "GET", f"/api/jobs/{job.id}/result")
    result = json.loads(data)
    assert result["source"] == "meeting.wav"
    assert [t["speaker"] for t in result["turns"]] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert result["summary"]["speakers"][0]["main_points"][0]["text"] == "Greets"

    status, headers, data = call(app, "GET", f"/api/jobs/{job.id}/files/meeting.md")
    assert status == 200 and "attachment" in headers["content-disposition"]
    assert "Greets" in data.decode()

    status, _, data = call(app, "GET", f"/api/jobs/{job.id}/audio")
    assert status == 200 and data == b"RIFF-fake"

    status, _, data = call(app, "GET", "/api/jobs")
    assert [j["id"] for j in json.loads(data)] == [job.id]


def test_extraction_switch_and_one_language(settings, factory, tmp_path):
    manager = web.JobManager(settings, tmp_path / "out", factory=factory)
    app = web.create_app(manager)
    status, _, data = upload(
        app, {"extract": "", "languages": "zh", "translate_to": "", "summary": ""}
    )
    assert status == 201
    assert wait(manager, json.loads(data)["id"]).state == "done"
    used_settings, options = factory.calls[0]
    assert used_settings.extraction == "off"
    assert options.languages == ["zh"] and options.translate_to is None and not options.summarize


def test_renaming_rewrites_the_files_and_survives_a_restart(settings, factory, tmp_path):
    manager = web.JobManager(settings, tmp_path / "out", factory=factory)
    app = web.create_app(manager)
    job = wait(manager, json.loads(upload(app)[2])["id"])

    body = json.dumps({"SPEAKER_00": " Alice "}).encode()
    status, _, data = call(
        app, "POST", f"/api/jobs/{job.id}/speakers", body, {"content-type": "application/json"}
    )
    assert status == 200
    assert json.loads(data)["speakers"][0]["name"] == "Alice"
    assert "### Alice" in (job.folder / "meeting.md").read_text()

    body = json.dumps({"SPEAKER_09": "Nobody"}).encode()
    assert call(app, "POST", f"/api/jobs/{job.id}/speakers", body)[0] == 400

    restarted = web.JobManager(settings, tmp_path / "out", factory=factory, start_worker=False)
    again = restarted.jobs[job.id]
    assert again.state == "done" and again.name == "meeting.wav"
    assert again.transcript is not None and again.transcript.speakers[0].name == "Alice"
    assert "meeting.md" in again.files


def test_bad_requests_get_clear_answers(settings, factory, tmp_path):
    manager = web.JobManager(settings, tmp_path / "out", factory=factory, start_worker=False)
    app = web.create_app(manager)

    status, _, data = upload(app, files=[])
    assert status == 400 and "Choose a recording" in json.loads(data)["error"]
    assert upload(app, {"speakers": "two"})[0] == 400
    assert upload(app, {"speakers": "99"})[0] == 400
    assert upload(app, {"translate_to": "en; rm -rf"})[0] == 400
    assert upload(app, {"languages": "klingon"})[0] == 400
    assert manager.jobs == {}

    assert call(app, "GET", "/api/jobs/nope")[0] == 404
    status, _, data = upload(app)
    job_id = json.loads(data)["id"]
    assert call(app, "GET", f"/api/jobs/{job_id}/result")[0] == 404  # not finished yet
    assert call(app, "GET", f"/api/jobs/{job_id}/files/..%2Fjob.json")[0] == 404


def test_failures_and_warnings_are_reported(settings, tmp_path):
    def broken(settings, options):  # type: ignore[no-untyped-def]
        raise RuntimeError("diarization model not found")

    manager = web.JobManager(settings, tmp_path / "out", factory=broken, start_worker=False)
    job = manager.new_job("talk.m4a", PipelineOptions(), extract=True)
    job.audio.write_bytes(b"x")
    manager.submit(job)
    manager.run_job(job)
    assert job.state == "error" and "diarization model not found" in (job.error or "")
    assert not (job.folder / web.JOB_FILE).exists()

    class Warns:
        progress = None

        def run(self, source, options):  # type: ignore[no-untyped-def]
            logging.getLogger("speakerlens.pipeline").warning("Translation failed: timeout")
            raise RuntimeError("stop")

    manager.factory = lambda s, o: Warns()  # type: ignore[assignment,misc,return-value]
    job = manager.new_job("talk.m4a", PipelineOptions(), extract=True)
    manager.run_job(job)
    assert job.warnings == ["Translation failed: timeout"]


def test_queue_position(settings, factory, tmp_path):
    manager = web.JobManager(settings, tmp_path / "out", factory=factory, start_worker=False)
    jobs = []
    for i in range(3):
        job = manager.new_job(f"m{i}.wav", PipelineOptions(), extract=True)
        job.created = float(i)
        manager.submit(job)
        jobs.append(job)
    jobs[0].state = "running"
    assert [manager.queue_position(j) for j in jobs] == [0, 1, 2]


def test_page_and_setup_info(settings, factory, tmp_path, monkeypatch):
    monkeypatch.setattr(web, "setup_problems", lambda s: ["HF_TOKEN is empty in .env."])
    app = web.create_app(
        web.JobManager(settings, tmp_path / "out", factory=factory, start_worker=False)
    )
    status, headers, data = call(app, "GET", "/")
    assert status == 200 and headers["content-type"].startswith("text/html")
    assert "Upload a recording" in data.decode()
    info = json.loads(call(app, "GET", "/api/info")[2])
    assert info["problems"] == ["HF_TOKEN is empty in .env."]
    assert info["llm"]["enabled"] is False and info["extract_default"] is True


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\talk.m4a", "talk.m4a"),
        ("会议 录音.m4a", "会议 录音.m4a"),
        (".hidden.wav", "hidden.wav"),
        ("a;b|c.wav", "a_b_c.wav"),
        ("", "recording"),
    ],
)
def test_safe_name(given, expected):
    assert web.safe_name(given) == expected

"""Chintu render worker — the 16 GB high-memory engine.

Runs inside a Hugging Face Docker Space (2 vCPU / 16 GB RAM, `cpu-basic`). The
Render brain holds no media code: it POSTs a composition here, this process
stages it, renders it with the real HyperFrames CLI (headless Chromium +
system FFmpeg), proves the artifact with ffprobe and hands the MP4 back over
HTTP so the brain can stream it to Telegram.

Endpoints
    GET  /health              readiness + hardware facts
    POST /render              {"html","project_slug","fps","quality","narration"}
                              -> 202 {"job_id","status":"queued"}
    GET  /status/<job_id>     {"status","percent","stage","probe","log_tail"}
    GET  /artifact/<job_id>   the finished MP4 (video/mp4)

Rendering is asynchronous on purpose: a Hugging Face Space proxies one
request with a short gateway timeout, so a multi-minute render cannot be a
single blocking POST. The caller polls /status and downloads /artifact.

Auth: set the Space secret `HF_RENDER_SECRET`; every endpoint except /health
then requires `Authorization: Bearer <secret>`.
"""
import asyncio
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "1.0.0"
HYPERFRAMES_VERSION = os.environ.get("HYPERFRAMES_VERSION", "0.8.160")
RENDER_TIMEOUT_SECONDS = int(os.environ.get("RENDER_TIMEOUT_SECONDS", "1500"))
SHARED_SECRET = os.environ.get("HF_RENDER_SECRET", "")
WORK_ROOT = os.environ.get("RENDER_WORK_ROOT", "/tmp/chintu-render")
MAX_JOBS_RETAINED = int(os.environ.get("MAX_JOBS_RETAINED", "8"))
NARRATION_VOICE = os.environ.get("NARRATION_VOICE", "en-US-AriaNeural")
ALLOWED_QUALITY = ("draft", "looks", "delivery")
PROJECT_SLUG_RE = re.compile(r"[^a-z0-9-]+")

JOBS = {}
JOBS_LOCK = threading.Lock()


# --------------------------------------------------------------------- helpers
def _mem_total_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _log(job, line):
    text = str(line).rstrip()
    if not text:
        return
    with JOBS_LOCK:
        job["log"].append(text[-4000:])
        job["log"] = job["log"][-250:]
    print(text, flush=True)


def _stage(job, percent, text):
    with JOBS_LOCK:
        job["percent"] = int(percent)
        job["stage"] = text
    _log(job, f"[{int(percent)}%] {text}")


def _tail(text, limit=4000):
    return (text or "")[-limit:]


def _run(job, cmd, cwd, timeout, label):
    """Run one command, echoing a bounded tail of its output into the job log."""
    _log(job, f"$ {' '.join(cmd)}")
    try:
        res = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label} exceeded {timeout}s")
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is not installed in this image: {exc}")
    if res.stdout:
        _log(job, res.stdout)
    if res.stderr:
        _log(job, res.stderr)
    return res


def _validate(payload):
    html = payload.get("html")
    if not isinstance(html, str) or not html.strip():
        return "html is required and must be a non-empty string"
    if "data-composition-id" not in html:
        return "composition has no element carrying data-composition-id"
    if "__timelines" not in html:
        return 'composition registers no timeline (window.__timelines["<id>"])'
    if "moviepy" in html:
        return "moviepy is forbidden in this pipeline"
    try:
        fps = int(payload.get("fps") or 30)
    except (TypeError, ValueError):
        return "fps must be an integer"
    if not 1 <= fps <= 240:
        return "fps must be between 1 and 240"
    quality = (payload.get("quality") or "looks").strip().lower()
    if quality not in ALLOWED_QUALITY:
        return f"quality must be one of {', '.join(ALLOWED_QUALITY)}"
    return None


def _probe(video):
    res = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
            "-show_entries", "format=duration,size", "-of", "json", video,
        ],
        capture_output=True, text=True, timeout=180,
    )
    data = json.loads(res.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    rate = str(stream.get("r_frame_rate") or "0/1")
    try:
        num, _, den = rate.partition("/")
        fps = round(float(num) / float(den or 1), 3)
    except Exception:
        fps = None
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "frames": int(stream.get("nb_read_frames") or 0) or None,
        "duration": round(float(fmt.get("duration") or 0), 3),
        "bytes": os.path.getsize(video),
    }


def _synthesize_narration(text, out_path):
    """edge-tts voice-over. Imported lazily so the worker boots without it."""
    try:
        import edge_tts
    except Exception as exc:  # pragma: no cover - depends on the image
        raise RuntimeError(
            f"narration was requested but edge-tts is unavailable ({exc}); "
            "install it in the Space image"
        )

    async def _speak():
        await edge_tts.Communicate(text, NARRATION_VOICE).save(out_path)

    asyncio.run(_speak())
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("edge-tts produced no audio")
    return out_path


def _mux(video, narration, out_path, job):
    # ffmpeg runs with cwd set to the video's directory, so every path it sees
    # must be absolute or a relative path would resolve inside that directory.
    video, narration, out_path = (os.path.abspath(p) for p in (video, narration, out_path))
    return _run(
        job,
        [
            "ffmpeg", "-y", "-i", video, "-i", narration,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_path,
        ],
        os.path.dirname(video),
        600,
        "ffmpeg mux",
    )


# ---------------------------------------------------------------------- render
def _render_job(job, payload):
    job_id = job["id"]
    try:
        slug = PROJECT_SLUG_RE.sub("-", str(payload.get("project_slug") or "chintu-motion").lower()).strip("-")
        slug = slug or "chintu-motion"
        fps = int(payload.get("fps") or 30)
        quality = (payload.get("quality") or "looks").strip().lower()
        narration_text = (payload.get("narration") or "").strip()
        project_dir = os.path.join(WORK_ROOT, job_id, slug)
        os.makedirs(os.path.join(project_dir, "renders"), exist_ok=True)

        _stage(job, 30, "staging composition and project config")
        with open(os.path.join(project_dir, "index.html"), "w", encoding="utf-8") as fh:
            fh.write(payload["html"])
        with open(os.path.join(project_dir, "hyperframes.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
                    "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
                    "paths": {
                        "blocks": "compositions",
                        "components": "compositions/components",
                        "assets": "assets",
                    },
                    "media": {"autoProxy": True},
                    "authoringSkill": "motion-graphics",
                },
                fh,
                indent=2,
            )
        _log(job, f"staged {len(payload['html'])} chars at {project_dir}")

        hf = ["npx", "--yes", f"hyperframes@{HYPERFRAMES_VERSION}"]
        _stage(job, 40, "verifying headless Chromium and the HyperFrames CLI")
        _run(job, hf + ["browser", "ensure"], project_dir, 600, "browser ensure")
        _run(job, hf + ["doctor", "--json"], project_dir, 300, "doctor")
        _run(job, hf + ["lint", "."], project_dir, 600, "lint")

        _stage(job, 80, "rendering frames (headless Chromium) and encoding with FFmpeg")
        render = _run(
            job,
            hf
            + [
                "render", ".", "--skill=motion-graphics",
                "-f", str(fps), "-q", quality, "-o", "./renders/video.mp4",
            ],
            project_dir,
            RENDER_TIMEOUT_SECONDS,
            "hyperframes render",
        )
        if render.returncode != 0:
            raise RuntimeError(f"hyperframes render exited {render.returncode}")
        video = os.path.join(project_dir, "renders", "video.mp4")
        if not os.path.exists(video) or os.path.getsize(video) == 0:
            raise RuntimeError("hyperframes render wrote no MP4")

        _stage(job, 92, "probing the MP4 frame count")
        probe = _probe(video)
        _log(job, f"ffprobe: {json.dumps(probe, sort_keys=True)}")
        if not probe.get("frames"):
            raise RuntimeError("the MP4 contains no decodable video frames")

        final = video
        if narration_text:
            _stage(job, 95, "synthesizing voice-over (edge-tts) and muxing audio")
            audio = os.path.join(project_dir, "narration.mp3")
            _synthesize_narration(narration_text, audio)
            final = os.path.join(project_dir, "renders", "final.mp4")
            _mux(video, audio, final, job)
            probe["audio"] = True
        else:
            _stage(job, 95, "finalizing MP4 muxing")

        with open(final, "rb") as fh:
            artifact = fh.read()
        with JOBS_LOCK:
            job.update(
                status="completed",
                percent=100,
                stage="complete",
                artifact=artifact,
                probe=probe,
                bytes=len(artifact),
                finished=time.time(),
            )
        _log(job, f"[100%] complete — {len(artifact)} bytes, {probe}")
    except Exception as exc:
        detail = f"{exc}\n{traceback.format_exc()}"
        with JOBS_LOCK:
            job.update(status="failed", stage="failed", error=str(exc), finished=time.time())
        _log(job, f"[FAILED] {detail}")


def _evict_old_jobs():
    with JOBS_LOCK:
        finished = [j for j in JOBS.values() if j["status"] in ("completed", "failed")]
        finished.sort(key=lambda j: j.get("finished") or 0)
        for job in finished[: max(0, len(finished) - MAX_JOBS_RETAINED)]:
            JOBS.pop(job["id"], None)
            shutil.rmtree(os.path.join(WORK_ROOT, job["id"]), ignore_errors=True)


# ------------------------------------------------------------------------ http
class Handler(BaseHTTPRequestHandler):
    server_version = f"ChintuRender/{VERSION}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the Space log readable
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status, body, content_type, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _authorised(self):
        if not SHARED_SECRET:
            return True
        header = self.headers.get("Authorization") or ""
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not token:
            token = (self.headers.get("X-Render-Secret") or "").strip()
        return bool(token) and hmac.compare_digest(token, SHARED_SECRET)

    def _guard(self):
        if self._authorised():
            return True
        self._json(401, {"error": "unauthorized", "detail": "send Authorization: Bearer <HF_RENDER_SECRET>"})
        return False

    def _job_id(self):
        return (self.path.rstrip("/").split("/")[-1] or "").strip()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/health":
            return self._json(
                200,
                {
                    "ok": True,
                    "service": "chintu-render-space",
                    "version": VERSION,
                    "role": "16 GB high-memory render engine",
                    "hardware": {"cpus": os.cpu_count(), "ram_mb": _mem_total_mb()},
                    "toolchain": {
                        "node": shutil.which("node") is not None,
                        "npx": shutil.which("npx") is not None,
                        "ffmpeg": shutil.which("ffmpeg") is not None,
                        "ffprobe": shutil.which("ffprobe") is not None,
                        "hyperframes": HYPERFRAMES_VERSION,
                    },
                    "jobs": {
                        "retained": len(JOBS),
                        "running": sum(1 for j in JOBS.values() if j["status"] == "running"),
                    },
                },
            )
        if path == "/":
            return self._send(200, b"Chintu render worker. GET /health\n", "text/plain")
        if path.startswith("/status/"):
            if not self._guard():
                return
            with JOBS_LOCK:
                job = JOBS.get(self._job_id())
                snapshot = (
                    {
                        "job_id": job["id"],
                        "status": job["status"],
                        "percent": job["percent"],
                        "stage": job["stage"],
                        "bytes": job.get("bytes"),
                        "probe": job.get("probe"),
                        "error": job.get("error"),
                        "log_tail": _tail("\n".join(job["log"])),
                    }
                    if job
                    else None
                )
            if snapshot is None:
                return self._json(404, {"error": "unknown job id"})
            return self._json(200, snapshot)
        if path.startswith("/artifact/"):
            if not self._guard():
                return
            with JOBS_LOCK:
                job = JOBS.get(self._job_id())
                artifact = job.get("artifact") if job else None
                complete = bool(job and job["status"] == "completed")
            if not complete or not artifact:
                return self._json(409, {"error": "artifact is not ready"})
            return self._send(
                200,
                artifact,
                "video/mp4",
                {"Content-Disposition": f'attachment; filename="{self._job_id()}.mp4"'},
            )
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path != "/render":
            return self._json(404, {"error": "not found"})
        if not self._guard():
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 48 * 1024 * 1024:
            return self._json(400, {"error": "a JSON body under 48 MB is required"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            return self._json(400, {"error": f"invalid JSON body: {exc}"})
        if not isinstance(payload, dict):
            return self._json(400, {"error": "body must be a JSON object"})
        problem = _validate(payload)
        if problem:
            return self._json(400, {"error": problem})

        _evict_old_jobs()
        job = {
            "id": uuid.uuid4().hex[:16],
            "status": "running",
            "percent": 15,
            "stage": "accepted",
            "created": time.time(),
            "log": [],
            "artifact": None,
            "probe": None,
        }
        with JOBS_LOCK:
            JOBS[job["id"]] = job
        _log(job, f"job {job['id']} accepted: {len(payload['html'])} chars, {payload.get('fps') or 30} fps")
        threading.Thread(target=_render_job, args=(job, payload), daemon=True).start()
        return self._json(
            202,
            {
                "job_id": job["id"],
                "status": "running",
                "poll": f"/status/{job['id']}",
                "artifact": f"/artifact/{job['id']}",
            },
        )


def main():
    os.makedirs(WORK_ROOT, exist_ok=True)
    port = int(os.environ.get("PORT", "7860"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(
        f"[chintu-render-space] v{VERSION} listening on 0.0.0.0:{port} "
        f"cpus={os.cpu_count()} ram={_mem_total_mb()}MB hyperframes={HYPERFRAMES_VERSION}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

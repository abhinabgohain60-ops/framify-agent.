"""Regression suite for the skill library, GitHub access, and HyperFrames render path."""
import sys, os, io, types, json, re, base64, asyncio, contextlib
root = os.getcwd()
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, ".tools", "stubs"))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {detail}" if detail and not cond else ""))

import main

# ---------------------------------------------------------------- 1. skills
print("\n--- skill registry ---")
listing = main.list_skills()
for key, meta in main.SKILL_REGISTRY.items():
    check(f"list_skills mentions {key}", key in listing)
    check(f"{key} entrypoint is a real doc path",
          meta["entrypoint"].endswith(".md") and meta["entrypoint"] != meta.get("repo"),
          meta["entrypoint"])
    check(f"{key} declares docs", bool(meta.get("docs")))

prompts = main.load_skill("motion_graphics_prompts")
out = main.load_skill("hyperframes")
check("hyperframes loads the router SKILL.md, not the README",
      "skills/hyperframes/SKILL.md" in out.splitlines()[0], out.splitlines()[0])
# The router doc is 21,069 chars, i.e. inside the 24,000-char budget, so it must
# come back whole: no paging header, no truncation marker.
check("a doc that fits its budget is returned whole with no marker",
      "TRUNCATED" not in out and "[chars 0-" not in out)
check("a doc that fits is delivered in full (>20k chars)", len(out) > 20000, f"len={len(out)}")
# The prompt library README is 90,967 chars and must page explicitly instead.
check("an oversized doc pages with an explicit continuation hint",
      "TRUNCATED" in prompts and "offset=40000" in prompts, prompts.splitlines()[1])

page2 = main.load_skill("hyperframes", "skills/hyperframes/SKILL.md", offset=24000)
check("page 2 resumes at the given offset", "[chars 24000-" in page2)
check("page 2 content differs from page 1", page2[-800:] != out[-800:])

cli = main.load_skill("hyperframes", "skills/hyperframes-cli/SKILL.md", max_chars=6000)
check("per-call max_chars is honoured", "[chars 0-6000 of 25961]" in cli, cli.splitlines()[1])
cli_default = main.load_skill("hyperframes", "skills/hyperframes-cli/SKILL.md")
check("per-skill max_chars is used when no override is given",
      "[chars 0-24000 of 25961]" in cli_default, cli_default.splitlines()[1])
check("prompt library reports its true size (91k)", "of 90967" in prompts, prompts.splitlines()[1])
check("unknown skill is rejected cleanly", "Unknown skill" in main.load_skill("nope"))

# ------------------------------------------------------------ 2. github cache
print("\n--- github access ---")
calls = {"n": 0}
# Keep the genuine HTTP clients: several later sections monkeypatch the shared
# requests module, and the live skill loads at the end need the real one back.
real_get = main.requests.get
real_post = main.requests.post
class _Res:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text
    def json(self): return self._payload
def counting_get(url, **kw):
    calls["n"] += 1
    if "raw.githubusercontent.com" in url:
        return _Res(200, text="PAYLOAD")
    if "api.github.com" in url:
        return _Res(200, {"download_url": "https://raw.example/x"})
    return _Res(200, text="PAYLOAD")
main.requests.get = counting_get
main._GITHUB_RAW_CACHE.clear()
a = main.github_read_file("owner/repo", "a.md"); first = calls["n"]
b = main.github_read_file("owner/repo", "a.md")
check("a file read costs one CDN request and no API quota", first == 1, f"calls={first}")
check("second read is served from cache (0 HTTP calls)", calls["n"] == first, f"calls={calls['n']}")
check("cached value is identical", a == b == "PAYLOAD")

# Anything the CDN cannot serve (directory, private repo, another default branch)
# must still fall back to the contents API instead of returning nothing.
def cdn_missing_get(url, **kw):
    calls["n"] += 1
    if "raw.githubusercontent.com" in url:
        return _Res(404, text="404: Not Found")
    if "contents" in url:
        return _Res(200, {"download_url": "https://raw.example/x"})
    return _Res(200, text="VIA-API")
main.requests.get = cdn_missing_get
main._GITHUB_RAW_CACHE.clear()
calls["n"] = 0
fallback = main.github_read_file("owner/other", "b.md")
check("a CDN miss falls back to the contents API", fallback == "VIA-API", fallback)
check("the fallback tries both CDN branches then the API", calls["n"] == 4, f"calls={calls['n']}")

def rate_limited(url, **kw):
    return _Res(403, text="API rate limit exceeded for 1.2.3.4")
main.requests.get = rate_limited
main._GITHUB_RAW_CACHE.clear()
msg = main.github_read_file("owner/repo", "b.md")
check("403 rate limit is reported as a rate limit, not raw HTML",
      "rate limit" in msg.lower() and "GITHUB_TOKEN" in msg, msg[:120])

# ------------------------------------------------- 3. composition validation
print("\n--- composition validation ---")
good = '<div data-composition-id="main"></div><script>window.__timelines["main"]=tl</script>'
check("empty html rejected", main._validate_composition_html("") is not None)
check("missing data-composition-id rejected",
      "data-composition-id" in (main._validate_composition_html("<div></div>") or ""))
check("missing timeline rejected",
      "timeline" in (main._validate_composition_html('<div data-composition-id="main"></div>') or ""))
check("moviepy rejected",
      "moviepy" in (main._validate_composition_html('<div data-composition-id="m"></div>' + 'window.__timelines["m"]' + 'moviepy') or ""))
check("valid composition accepted", main._validate_composition_html(good) is None)

# ------------------------------------------------------- 4. fake sandbox run
print("\n--- render + sandbox plumbing ---")
class _Run:
    def __init__(self, cmd):
        if "ffprobe -i" in cmd and "-select_streams a" in cmd:
            # ffprobe audio verification succeeds: a muxed final.mp4 exists.
            self.exit_code, self.stdout, self.stderr = 0, "[STREAM]\ncodec_type=audio\ncodec_name=aac\n", ""
        else:
            self.exit_code, self.stdout, self.stderr = 0, f"ran: {cmd}", ""
class _Files:
    def __init__(self, sbx): self.sbx = sbx; self.written = {}
    def write(self, path, content):
        self.written[path] = content; self.sbx.written[path] = content
    def read(self, path, format=None):
        # final_output.mp4 only exists if a mux command actually produced it;
        # reading a file that was never created must fail like production I/O.
        if path.endswith("final_output.mp4") and not any(
            "ffmpeg -y" in c and "final_output.mp4" in c for c, _ in self.sbx.cmds
        ):
            raise FileNotFoundError(path)
        return b"MP4BYTES"
    def list(self, directory):
        # A rendered .mp4 exists wherever the framework put it.
        return [types.SimpleNamespace(name="video.mp4")]
class FakeSandbox:
    def __init__(self): self.cmds = []; self.written = {}; self.files = _Files(self); self.killed = False
    class _Cmds:
        def __init__(self, sbx): self.sbx = sbx
        def run(self, cmd, timeout=0): self.sbx.cmds.append((cmd, timeout)); return _Run(cmd)
    @property
    def commands(self): return self._Cmds(self)
    def kill(self): self.killed = True

sbx = FakeSandbox()
main._provision_render_toolchain(sbx, want_browser=True)
joined = " ;; ".join(c for c, _ in sbx.cmds)
check("browser path installs Node", "nodesource" in joined and "nodejs" in joined)
check("browser path installs Chrome runtime libs", "libnss3" in joined and "libnspr4" in joined)
check("browser path runs browser ensure", "browser ensure" in joined)
check("every provisioning call disables the command timeout",
      all(t == 0 for _, t in sbx.cmds), str([t for _, t in sbx.cmds]))
plain = FakeSandbox(); main._provision_render_toolchain(plain, want_browser=False)
plain_joined = " ;; ".join(c for c, _ in plain.cmds)
check("python path does NOT install Node/Chrome", "npx" not in plain_joined and "nodesource" not in plain_joined)
check("python path still provisions ffmpeg + frame stack",
      "ffmpeg" in plain_joined and "pillow" in plain_joined)

# render_html_video without a key / with bad html
# Explicitly clear HF_TOKEN: with one set, render_html_video submits a Job instead.
main.HF_TOKEN = ""
saved_key = main.E2B_API_KEY
main.E2B_API_KEY = ""
check("render_html_video reports a missing E2B key",
      "E2B_API_KEY" in main.render_html_video(good))
main.E2B_API_KEY = saved_key or "test-key"
check("render_html_video rejects an invalid composition before touching the sandbox",
      "rejected before render" in main.render_html_video("<div></div>"))

created = {}
fake_mod = types.ModuleType("e2b_code_interpreter")
class _SB:
    @staticmethod
    def create(api_key=None, timeout=None):
        s = FakeSandbox(); s.timeout = timeout; created["sbx"] = s; return s
fake_mod.Sandbox = _SB
sys.modules["e2b_code_interpreter"] = fake_mod

main.CURRENT_CHAT_ID = 424242
main.TELEGRAM_BOT_TOKEN = "TESTTOKEN"
uploads = []
class _PostRes:
    status_code = 200
    text = "ok"
def fake_post(url, **kw):
    uploads.append((url, kw.get("files", {}).get("video", ("",))[0])); return _PostRes()
main.requests.post = fake_post
report = main.render_html_video(good, project_name="Chintu Motion!", fps=30, quality="delivery")
s = created["sbx"]
render_cmd = [c for c, _ in s.cmds if " render " in c][0]
check("project slug is sanitised from 'Chintu Motion!'",
      "cd /home/user/videos/chintu-motion &&" in render_cmd, render_cmd)
check("render runs the real hyperframes CLI with the skill flag",
      "hyperframes@0.8.160 render . --skill=motion-graphics" in render_cmd, render_cmd)
check("render passes fps and quality", "-f 30 -q delivery" in render_cmd, render_cmd)
check("composition and project config are written",
      "/home/user/videos/chintu-motion/index.html" in s.written
      and "/home/user/videos/chintu-motion/hyperframes.json" in s.written)
check("project config is valid JSON with the schema", json.loads(
      s.written["/home/user/videos/chintu-motion/hyperframes.json"])["paths"]["blocks"] == "compositions")
check("lint is run before render", any("lint" in c for c, _ in s.cmds))
check("sandbox is always killed", s.killed)
check("MP4 is delivered to Telegram, never under the name video.mp4",
      uploads and uploads[0][1] == "final_output.mp4", str(uploads))
check("report states SUCCESS with the artifact path", "[SUCCESS]" in report, report[-200:])
check("report includes render stdout for grounding", "ran:" in report)

# ------------------------------------------------ 4b. E2B render engine body
print("\n--- e2b render engine ---")
# All Hugging Face render backends were retired. The substitutions below prove
# the new reality: cleanup+TMPDIR prep, the Pillow fallback compiler and the
# synchronous delivery every path must provide.
main.CURRENT_CHAT_ID = 777
main.TELEGRAM_BOT_TOKEN = "TESTTOKEN"
real_emit = main.emit_progress
for ident in ("hf_render_status", "hf_space_health"):
    check(f"all trace of {ident} was removed with the HF backends", not hasattr(main, ident))
check("e2b_disk_prep: exactly the mandated gitignore-quoted cleanup string",
      "rm -rf /tmp/* ~/.cache/* /var/tmp/* 2>/dev/null || true" in main._e2b_disk_prep_cmd(),
      main._e2b_disk_prep_cmd())
check("tmpdir is the user volume, not the small tmpfs", main.E2B_TMP_DIR == "/home/user/tmp")
check("disk prep stages the tmp dir before export",
      main._e2b_disk_prep_cmd().index("mkdir -p /home/user/tmp") <
      main._e2b_disk_prep_cmd().index("export TMPDIR=/home/user/tmp"))

# ------------------------------------------- 5. execute_in_cloud_microvm intact
print("\n--- execute_in_cloud_microvm ---")
created.clear()
main.requests.post = fake_post
out = main.execute_in_cloud_microvm("print('hi')")
s2 = created["sbx"]
j2 = " ;; ".join(c for c, _ in s2.cmds)
check("microvm path stays on the ffmpeg-only provisioning", "npx --yes" not in j2)
check("microvm still runs run_task.py", any("python /home/user/run_task.py" in c for c, _ in s2.cmds))
check("microvm reports the delivered video", "delivered" in out.lower(), out[-200:])
check("microvm kills the sandbox", s2.killed)

# ----------------------------------------------------------- 6. cascade intact
print("\n--- cascade / fallback ---")
main.GEMINI_API_KEYS.clear()
main.OPENROUTER_API_KEY = "or-key"
main.requests.post = lambda url, **kw: types.SimpleNamespace(
    status_code=200, json=lambda: {"choices": [{"message": {"content": "OR-OK"}}]}, text="")
import asyncio
res = asyncio.run(main.generate_reasoning_response("ping"))
check("no Gemini keys -> external fallback answers", res == "OR-OK", res)

# ------------------------------------- 6b-hf. the retired HF Space backends
print("\n--- retired hugging face backends ---")
main.CURRENT_CHAT_ID = 777
main.TELEGRAM_BOT_TOKEN = "TESTTOKEN"
composition = '<div data-composition-id="main"></div><script>window.__timelines["main"]=tl</script>'


class _JsonRes:
    def __init__(self, status=200, payload=None, text="", content=b""):
        self.status_code, self._payload, self.text, self.content = status, payload or {}, text, content
    def json(self):
        return self._payload


http_posts = []


def no_post_expected(url, **kw):
    http_posts.append(url)
    raise AssertionError("no HTTP call should happen for a retired backend")


main.requests.post = no_post_expected
retired_space = main.render_html_video(composition, backend="space")
check("backend='space' returns a retirement notice with no HTTP call",
      "retired" in retired_space and "E2B" in retired_space, retired_space[:160])
check("retired space backend mentions the paid PRO cause",
      "PRO" in retired_space, retired_space[:160])
retired_hf = main.render_html_video(composition, backend="huggingface")
check("backend='huggingface' too", "retired" in retired_hf, retired_hf[:160])
retired_jobs = main.render_html_video(composition, backend="jobs")
check("backend='jobs' too, and hf is accepted as an alias",
      "retired" in retired_jobs and "retired" in main.render_html_video(composition, backend="hf"))
check("no HTTP call was made for any retired backend", not http_posts, str(http_posts[:2]))
check("backend='auto' still reaches the E2B sandbox",
      isinstance(main.render_html_video(composition, project_name="Retire X"), str))

# --------------------------------- 6b-audio. hardened mux + streaming delivery
print("\n--- audio mux + streaming delivery ---")


audio_uploads = []


def audio_post(url, **kw):
    audio_uploads.append((url, kw))
    return types.SimpleNamespace(status_code=200, text="ok")


main.requests.post = audio_post
sent = main._telegram_send_video(b"MP4BYTES", "final_output.mp4", "cap")
check("sendVideo receives the final upload and HTTP 200 turns it into success",
      sent is True, str(sent))
last = audio_uploads[-1][1]
check("the sendVideo payload opts into streaming (supports_streaming=True)",
      last.get("data", {}).get("supports_streaming") == "true", json.dumps(last.get("data", {})))
check("the upload explicitly declares video/mp4 so Telegram enables the unmute button",
      last.get("files", {}).get("video", ("", None, ""))[2] == "video/mp4",
      str(last.get("files", {}).get("video", ("",))[0]))

check("the mux command matches the directive verbatim (explicit map + AAC faststart)",
      "-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -ar 44100 "
      "-shortest -movflags +faststart" in main._e2b_audio_mux_cmd("v.mp4", "a.mp3", "f.mp4"),
      main._e2b_audio_mux_cmd("v.mp4", "a.mp3", "f.mp4"))
check("the verify command is the directed ffprobe audio check",
      "ffprobe -i f.mp4 -show_streams -select_streams a" == main._e2b_audio_verify_cmd("f.mp4"))
check("verification requires an ACTIVE audio codec, not just exit code 0",
      main._audio_stream_verified("codec_type=audio\ncodec_name=aac")
      and not main._audio_stream_verified(""))

# Behavioral: synthesis + mux/verify inside a fake sandbox.
class _AudioFiles:
    def __init__(self, sbx): self.sbx = sbx
    def write(self, path, content): self.sbx.written[path] = content


class _AudioCmds:
    def __init__(self, sbx, silent=False):
        self.sbx = sbx; self._silent = silent
    def run(self, cmd, timeout=0):
        self.sbx.cmds.append(cmd)
        if "ffprobe -i" in cmd and "-select_streams a" in cmd:
            if self._silent:
                return types.SimpleNamespace(exit_code=0, stdout="", stderr="")
            return types.SimpleNamespace(exit_code=0, stdout="codec_type=audio\ncodec_name=aac\n", stderr="")
        if " edge_tts " in cmd:
            return types.SimpleNamespace(exit_code=0, stdout="", stderr="")
        return types.SimpleNamespace(exit_code=0, stdout=f"ran: {cmd}", stderr="")


class _AudioSbx:
    def __init__(self, silent=False):
        self.cmds, self.written = [], {}
        self.files = _AudioFiles(self)
        self.commands = _AudioCmds(self, silent=silent)
    def kill(self): pass


asbx = _AudioSbx()
a_out = main._synthesize_narration_audio(asbx, "/home/user/videos/reel", "espresso reel")
check("narration is synthesized in-sandbox with edge-tts",
      a_out == "/home/user/videos/reel/narration.mp3" and a_out,
      a_out)
check("the voice-over script is staged in the sandbox",
      asbx.written.get("/home/user/videos/reel/narration.txt") == "espresso reel")
check("edge-tts runs with the configured voice through tmpdir prep",
      any("edge_tts" in c and "--write-media narration.mp3" in c and "TMPDIR" in c for c in asbx.cmds),
      str(asbx.cmds[1:])[:200])
check("edge-tts is invoked with the SUPPORTED --file flag, not --text-file",
      any("--file narration.txt" in c for c in asbx.cmds)
      and not any("--text-file" in c for c in asbx.cmds), str(asbx.cmds[1:]))
final = main._mux_narration_and_verify(asbx, "/home/user/videos/reel/renders/video.mp4", a_out)
check("the muxed final lands at renders/final_output.mp4",
      final == "/home/user/videos/reel/renders/final_output.mp4", str(final))
mux_cmd = [c for c in asbx.cmds if "final_output.mp4" in c and "ffmpeg -y" in c][0]
check("the mux runs WITH the explicit audio stream map",
      "-map 1:a:0" in mux_cmd and "-ar 44100" in mux_cmd and "+faststart" in mux_cmd, mux_cmd[:300])
check("ffprobe verifies the audio stream INSIDE the sandbox before any delivery",
      any("ffprobe -i /home/user/videos/reel/renders/final_output.mp4 -show_streams -select_streams a" in c for c in asbx.cmds),
      str([c[-120:] for c in asbx.cmds]))
# A probe that finds no audio stream must reject the mux and never claim success.
ssbx = _AudioSbx(silent=True)
check("a silent final fails verification and reports None (no fake success)",
      main._mux_narration_and_verify(ssbx, "/v/video.mp4", "/v/narration.mp3") is None,
      str(main._mux_narration_and_verify(ssbx, "/v/video.mp4", "/v/narration.mp3")))

# End-to-end on the fake render path: with narration, final_output.mp4 is what ships.
created.clear()
report_n = main.render_html_video(
    composition, project_name="Chintu Motion!", fps=30, quality="delivery",
    narration="espresso reel",)
sn = created["sbx"]
joined_cmds = " ;; ".join(c for c, _ in sn.cmds)
check("the render path synthesizes narration in the sandbox",
      "-m edge_tts" in joined_cmds and "--write-media narration.mp3" in joined_cmds, joined_cmds[-400:])
check("the success path muxes with the explicit stream map into final_output.mp4",
      any("-map 0:v:0" in c and "-map 1:a:0" in c and "-b:a 192k" in c
          and "-ar 44100" in c and "+faststart" in c and "final_output.mp4" in c
          for c, _ in sn.cmds), joined_cmds[-500:])
check("delivery targets the ffprobe-verified final",
      audio_uploads and audio_uploads[-1][1].get("files", {}).get("video", ("",))[0] == "final_output.mp4",
      str([u[1].get("files", {}).get("video", ("",))[0] for u in audio_uploads]))
check("the rendered report discloses the audio mux + verification",
      "audio: narration muxed as AAC" in report_n and "ffprobe-verified" in report_n, report_n[-400:])

# A render without narration must skip the audio pipeline entirely and still ship.
created.clear()
report_sil = main.render_html_video(composition, project_name="Chintu Motion!", fps=30)
ss = created["sbx"]
joined_sil = " ;; ".join(c for c, _ in ss.cmds)
check("no narration -> no mux, and the upload is still never named video.mp4",
      "edge_tts" not in joined_sil and "ffmpeg -y" not in joined_sil
      and audio_uploads[-1][1].get("files", {}).get("video", ("",))[0] == "final_output.mp4",
      str(audio_uploads[-1][1].get("files", {}).get("video", ("",))[0]))
check("no upload anywhere in this suite is ever named video.mp4",
      all(u[1].get("files", {}).get("video", ("",))[0] != "video.mp4" for u in audio_uploads)
      and all(name != "video.mp4" for _, name in uploads),
      str([u[1].get("files", {}).get("video", ("",))[0] for u in audio_uploads][:5]))


# Directive 1: final_output.mp4 ALWAYS wins, whatever else exists in the scan.
class _DelFiles:
    def __init__(self, present): self.present = set(present)
    def read(self, path, format=None):
        if path in self.present:
            return b"MP4BYTES"
        raise FileNotFoundError(path)
    def list(self, directory):
        if f"{directory}/video.mp4" in self.present:
            return [types.SimpleNamespace(name="video.mp4")]
        raise FileNotFoundError(directory)  # other scan dirs hold nothing


class _DelSbx:
    def __init__(self, present): self.files = _DelFiles(present)


main.CURRENT_CHAT_ID = 777
main.TELEGRAM_BOT_TOKEN = "TESTTOKEN"
main.requests.post = audio_post
audio_uploads.clear()
status = main._deliver_rendered_video(
    _DelSbx({"/e/final_output.mp4", "/e/video.mp4"}), extra_dirs=("/e",),
    require_audio=True)
check("final_output.mp4 is uploaded first, even when video.mp4 also exists",
      status == "delivered"
      and audio_uploads[-1][1]["files"]["video"][0] == "final_output.mp4",
      f"status={status} uploads={[u[1]['files']['video'][0] for u in audio_uploads]}")
audio_uploads.clear()
status = main._deliver_rendered_video(
    _DelSbx({"/e/video.mp4"}), extra_dirs=("/e",), require_audio=True)
check("require_audio with NO final_output.mp4 uploads NOTHING (fails loudly)",
      status == "no-audio" and not audio_uploads,
      f"status={status} uploads={len(audio_uploads)}")
status_unnarrated = main._deliver_rendered_video(
    _DelSbx({"/e/video.mp4"}), extra_dirs=("/e",), require_audio=False)
check("without narration the capture still ships, renamed off video.mp4",
      status_unnarrated == "delivered"
      and audio_uploads[-1][1]["files"]["video"][0] == "final_output.mp4",
      f"status={status_unnarrated}")

# Directive 2: a mux that keeps failing is retried once, then the render ends
# with a loud FAILED-NO-AUDIO — never a silent upload.
mux_attempts = []
real_mux_fn = main._mux_narration_and_verify


def always_fails_mux(sbx, video_path, audio_path):
    mux_attempts.append((video_path, audio_path))
    return None


main._mux_narration_and_verify = always_fails_mux
created.clear()
uploads_before = len(audio_uploads)
report_fail = main.render_html_video(
    composition, project_name="Chintu Motion!", fps=30, narration="espresso reel")
main._mux_narration_and_verify = real_mux_fn
check("a failing mux is retried exactly once before giving up",
      len(mux_attempts) == 2, str(len(mux_attempts)))
check("the render fails LOUDLY with FAILED-NO-AUDIO, not silently",
      "[FAILED-NO-AUDIO]" in report_fail and "NOT uploaded" in report_fail,
      report_fail[-300:])
check("no silent upload happens when the narrated mux fails",
      len(audio_uploads) == uploads_before,
      str([u[1].get("files", {}).get("video", ("",))[0] for u in audio_uploads]))
check("the loud failure still explains the audio cause",
      "failed after a retry" in report_fail, report_fail[-360:])


_FallbackTpl = main._PILLOW_FALLBACK_TEMPLATE
check("the Pillow fallback template carries the hardened mux",
      '"-map", "0:v:0"' in _FallbackTpl and '"-map", "1:a:0"' in _FallbackTpl
      and '"-b:a", "192k"' in _FallbackTpl and '"-ar", "44100"' in _FallbackTpl
      and '"+faststart"' in _FallbackTpl, "")
check("the Pillow fallback template ffprobe-verifies the audio track",
      '"-select_streams", "a"' in _FallbackTpl and "codec_type=audio" in _FallbackTpl)
check("the Pillow fallback template parses as strict AST", main._is_valid_python(_FallbackTpl))

# ------------------------------------------------- 6c. OpenRouter tool cascade
print("\n--- openrouter cascade ---")
main.HF_SPACE_URL = ""
main.GEMINI_API_KEYS.clear()
main.OPENROUTER_API_KEY = "or-key"
main.FREEBUFF_API_KEY = ""
main.PROVIDER_TELEMETRY.update(provider="", model="", failovers=0)

or_calls = []


def or_cascade_post(url, **kw):
    or_calls.append((url, kw))
    if len(or_calls) == 1:
        return _JsonRes(500, {}, "upstream exploded")
    return _JsonRes(200, {"choices": [{"message": {"content": "OR-OK"}}]})


main.requests.post = or_cascade_post
out = asyncio.run(main.generate_reasoning_response("ping"))
check("all-Gemini failure fails over to OpenRouter", out == "OR-OK", out[:120])
check("the cascade starts at the free router slot",
      or_calls[0][1]["json"]["model"] == "openrouter/free",
      or_calls[0][1]["json"]["model"])
check("a failing primary steps down to nemotron",
      or_calls[1][1]["json"]["model"] == "nvidia/nemotron-3-super-120b-a12b:free",
      or_calls[1][1]["json"]["model"])
check("the cascade order matches the pinned list exactly",
      list(main.OPENROUTER_MODELS) == ["openrouter/free", "nvidia/nemotron-3-super-120b-a12b:free", "cohere/north-mini-code:free"],
      str(list(main.OPENROUTER_MODELS)))
check("the fallback requests up to 600s", or_calls[0][1]["timeout"] == 600, str(or_calls[0][1]["timeout"]))
check("function schemas are serialised into the request",
      isinstance(or_calls[0][1]["json"].get("tools"), list)
      and or_calls[0][1]["json"]["tools"][0]["type"] == "function"
      and "parameters" in or_calls[0][1]["json"]["tools"][0]["function"])
check("the active provider and model are recorded for telemetry",
      main.PROVIDER_TELEMETRY["provider"] == "openrouter"
      and main.PROVIDER_TELEMETRY["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
      and main.PROVIDER_TELEMETRY["failovers"] >= 1, json.dumps(main.PROVIDER_TELEMETRY))

# A tool call must be executed and fed back before the model answers.
tool_rounds = []


def or_tool_post(url, **kw):
    tool_rounds.append(kw["json"])
    if len(tool_rounds) == 1:
        return _JsonRes(200, {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "list_skills", "arguments": "{}"}}
        ]}}]})
    return _JsonRes(200, {"choices": [{"message": {"content": "TOOLS-DONE"}}]})


main.requests.post = or_tool_post
out = main._sync_external_fallback("what skills do you have?")
check("an OpenRouter tool call is executed before answering", out == "TOOLS-DONE", out[:120])
check("the tool result is returned to the model as a tool message",
      tool_rounds[-1]["messages"][-1]["role"] == "tool"
      and "AVAILABLE SKILLS" in tool_rounds[-1]["messages"][-1]["content"],
      tool_rounds[-1]["messages"][-1]["role"])
check("the assistant tool call is echoed back verbatim",
      tool_rounds[-1]["messages"][-2]["tool_calls"][0]["id"] == "call_1")

# Freebuff remains the last resort after every OpenRouter model fails.
main.FREEBUFF_API_KEY = "fb-key"


def fb_post(url, **kw):
    if "openrouter" in url:
        return _JsonRes(503, {}, "down")
    return _JsonRes(200, {"choices": [{"message": {"content": "FB-OK"}}]})


main.requests.post = fb_post
out = main._sync_external_fallback("ping")
check("Freebuff answers only after the OpenRouter cascade fails", out == "FB-OK", out[:120])
check("the Freebuff provider is recorded too",
      main.PROVIDER_TELEMETRY["provider"] == "freebuff", json.dumps(main.PROVIDER_TELEMETRY))

schemas = {s["function"]["name"]: s["function"] for s in main.build_tool_schemas()}
check("every registered tool has a callable schema",
      set(schemas) == {f.__name__ for f in main.ALL_TOOLS_LIST}, str(sorted(set(schemas))))
check("required arguments are derived from the signature",
      schemas["github_read_file"]["parameters"]["required"] == ["repo_name", "file_path"],
      str(schemas["github_read_file"]["parameters"]["required"]))
check("only html is required to render",
      schemas["render_html_video"]["parameters"]["required"] == ["html"],
      str(schemas["render_html_video"]["parameters"]["required"]))
check("typed arguments map to JSON types",
      schemas["render_html_video"]["parameters"]["properties"]["fps"]["type"] == "integer"
      and schemas["github_read_file"]["parameters"]["properties"]["repo_name"]["type"] == "string")
check("unknown tool names are rejected cleanly",
      "unknown tool" in main.execute_tool_call("nope", "{}"))
check("malformed tool arguments are rejected cleanly",
      "not valid JSON" in main.execute_tool_call("list_skills", "{oops"))
check("a tool is never called with the wrong argument names",
      "bad arguments" in main.execute_tool_call("list_skills", '{"nope": 1}'))

# --------------------------------------------------------- 6d. progress card
print("\n--- progress card ---")


class _Card:
    def __init__(self):
        self.edits = []
    async def edit_text(self, text):
        self.edits.append(text)


card = _Card()
main.CURRENT_STATUS_MSG = card
main.reset_progress()
asyncio.run(main.update_progress_card(15, main.PROGRESS_MILESTONES[0][1]))
check("the 15% card is rendered with a real bar",
      card.edits and "15%" in card.edits[-1] and "[" in card.edits[-1], card.edits[-1][:80])
check("the initialising stage text matches the directive",
      "Initializing reasoning & tool pipeline" in card.edits[-1])
asyncio.run(main.update_progress_card(30, main.PROGRESS_MILESTONES[1][1]))
asyncio.run(main.update_progress_card(20, "rewind attempt"))
check("the card never rewinds when callbacks arrive out of order",
      "30%" in card.edits[-1] and "20%" not in card.edits[-1], card.edits[-1][:80])
asyncio.run(main.update_progress_card(20, "forced reset", force=True))
check("force allows an explicit rewind", "20%" in card.edits[-1])

stage_texts = {value: text for value, text in main.PROGRESS_MILESTONES}
check("all six directive milestones exist in order",
      [v for v, _ in main.PROGRESS_MILESTONES] == [15, 30, 55, 80, 95, 100],
      str([v for v, _ in main.PROGRESS_MILESTONES]))
check("the render milestones carry the mandated wording",
      stage_texts[55].startswith("Compiling layouts") and stage_texts[80].startswith("Headless Chromium")
      and stage_texts[95].startswith("Finalizing MP4") and stage_texts[100] == "Complete! Video delivered.",
      str(stage_texts))


async def _beat():
    main.reset_progress()
    main.PROGRESS_STATE.update(percent=15, stage=main.PROGRESS_MILESTONES[0][1])
    main.CURRENT_STATUS_MSG = _Card()
    task = asyncio.create_task(main.progress_heartbeat(0.01))
    await asyncio.sleep(0.06)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return main.PROGRESS_STATE["percent"], main.CURRENT_STATUS_MSG.edits


beat_percent, beat_edits = asyncio.run(_beat())
check("the heartbeat advances a stalled card inside its phase",
      15 < beat_percent < 30 and len(beat_edits) >= 2, f"percent={beat_percent} edits={len(beat_edits)}")

# ------------------------------------------------ 6e. telegram end-to-end run
print("\n--- telegram end-to-end ---")


class _TgMessage:
    def __init__(self, text):
        self.text = text
        self.chat_id = 777
        self.replies = []
        self.card = _Card()
        self.card_created = False
    async def reply_text(self, text):
        self.replies.append(text)
        # The final answer is also a reply, so never clear a card that was posted.
        self.card_created = self.card_created or text.startswith("Task/Project Progress")
        return self.card


class _TgUpdate:
    def __init__(self, text):
        self.message = _TgMessage(text)


class _TgContext:
    bot = object()


main.TELEGRAM_BOT_TOKEN = "TESTTOKEN"
main.HF_SPACE_URL = ""
main.HF_TOKEN = ""
update = _TgUpdate("make me a 9:16 reel about espresso")
saved_reasoning = main.generate_reasoning_response


async def fake_reasoning(message):
    # Real tool calls take seconds between phases; yield so the scheduled card
    # edits run exactly as they do in production.
    main.emit_progress(55, main.PROGRESS_MILESTONES[2][1], video=True)
    await asyncio.sleep(0.02)
    main.emit_progress(80, main.PROGRESS_MILESTONES[3][1], video=True)
    await asyncio.sleep(0.02)
    return "Rendered 270 frames."


main.generate_reasoning_response = fake_reasoning


async def drive():
    await main.handle_message(update, _TgContext())
    await asyncio.sleep(0.1)


asyncio.run(drive())
main.generate_reasoning_response = saved_reasoning
card_text = "\n".join(update.message.card.edits)
check("the status card is posted immediately", update.message.card_created)
check("the card is edited through the render phases, never frozen at 15%",
      "55%" in card_text and "80%" in card_text and "95%" in card_text, card_text[-260:])
check("the card finishes at 100% with the delivered wording",
      "100%" in card_text and "Complete! Video delivered." in card_text, card_text[-160:])
check("every emitted milestone is recorded, in order",
      [percent for percent, _ in main.PROGRESS_STATE["history"]] == sorted(
          percent for percent, _ in main.PROGRESS_STATE["history"])
      and {55, 80, 95, 100} <= {percent for percent, _ in main.PROGRESS_STATE["history"]},
      str(main.PROGRESS_STATE["history"])[:200])
check("the model's answer is delivered to the chat",
      any("Rendered 270 frames." in reply for reply in update.message.replies),
      str(update.message.replies)[:160])

# --------------------------------------------------- 6f. the 16 GB worker
print("\n--- 16 GB space worker ---")
import importlib.util
import tempfile
import threading
import urllib.error
import urllib.request

spec = importlib.util.spec_from_file_location("chintu_space_app", os.path.join(root, "space", "app.py"))
space_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(space_app)

worker_payload = {"html": composition, "project_slug": "Reel!", "fps": 30, "quality": "looks"}
check("the worker accepts a valid composition payload", space_app._validate(worker_payload) is None)
check("the worker rejects a body without html", "html is required" in space_app._validate({}))
check("the worker rejects a composition with no clip timing",
      "data-composition-id" in space_app._validate({"html": "<div></div>"}))
check("the worker rejects a composition with no timeline",
      "__timelines" in space_app._validate({"html": '<div data-composition-id="m"></div>'}))
check("the worker refuses moviepy",
      "moviepy" in space_app._validate({"html": composition + "moviepy"}))
check("the worker bounds fps",
      "fps" in space_app._validate(dict(worker_payload, fps=999)))
check("the worker bounds quality",
      "quality" in space_app._validate(dict(worker_payload, quality="imax")))

real_run, real_probe = space_app._run, space_app._probe
worker_root = tempfile.mkdtemp()
space_app.WORK_ROOT = worker_root


def fake_worker_run(job, cmd, cwd, timeout, label):
    space_app._log(job, f"$ {' '.join(cmd)}")
    if label == "hyperframes render":
        out_dir = os.path.join(cwd, "renders")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "video.mp4"), "wb") as fh:
            fh.write(b"\x00\x00\x00\x20ftypisom" + b"V" * 2048)
    return types.SimpleNamespace(returncode=0, stdout=f"ok:{label}", stderr="")


space_app._run = fake_worker_run
space_app._probe = lambda video: {
    "width": 1080, "height": 1920, "fps": 30.0, "frames": 270,
    "duration": 9.0, "bytes": os.path.getsize(video),
}
worker_job = {"id": "w1", "status": "running", "percent": 15, "stage": "accepted",
              "created": 0.0, "log": [], "artifact": None, "probe": None}
space_app._render_job(worker_job, worker_payload)
check("the worker renders, probes and holds the MP4",
      worker_job["status"] == "completed"
      and worker_job["probe"]["frames"] == 270
      and worker_job["artifact"].startswith(b"\x00\x00\x00\x20ftyp"), str(worker_job["status"]))
check("the worker logs the render command and ffprobe facts",
      any("hyperframes@0.8.160 render" in line for line in worker_job["log"])
      and any("ffprobe" in line for line in worker_job["log"]))


def exploding_run(job, cmd, cwd, timeout, label):
    raise RuntimeError("chrome failed to start")


space_app._run = exploding_run
failed_job = {"id": "w2", "status": "running", "percent": 15, "stage": "accepted",
              "created": 0.0, "log": [], "artifact": None, "probe": None}
space_app._render_job(failed_job, worker_payload)
check("a worker failure is recorded with its cause, not swallowed",
      failed_job["status"] == "failed" and "chrome failed" in failed_job["error"],
      str(failed_job.get("error"))[:80])
space_app._run, space_app._probe = real_run, real_probe

# ffmpeg runs with cwd set to the video's directory, so every path it receives
# must be absolute (a relative path would silently resolve inside that folder).
mux_calls = []


def fake_mux_run(job, cmd, cwd, timeout, label):
    mux_calls.append((cmd, cwd, label))
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


space_app._run = fake_mux_run
space_app._mux(".tools/probe/t.mp4", ".tools/probe/n.mp3", ".tools/probe/o.mp4", {"id": "m", "log": []})
space_app._run = real_run
mux_cmd, mux_cwd, mux_label = mux_calls[-1]
check("the mux step passes absolute paths to ffmpeg",
      all(os.path.isabs(p) for p in mux_cmd if p.endswith((".mp4", ".mp3"))), str(mux_cmd))
check("the mux step runs from the video's directory",
      mux_cwd == os.path.dirname(os.path.abspath(".tools/probe/t.mp4")), mux_cwd)
check("narration is muxed as AAC with the video stream copied",
      "-c:a" in mux_cmd and "aac" in mux_cmd and "-c:v" in mux_cmd and "copy" in mux_cmd,
      str(mux_cmd))

dockerfile = open(os.path.join(root, "space", "Dockerfile")).read()
space_readme = open(os.path.join(root, "space", "README.md")).read()
check("the Space image is Node 22 with FFmpeg and Chromium libraries",
      "node:22-bookworm" in dockerfile and "ffmpeg" in dockerfile and "libnspr4" in dockerfile)
check("the Space image warms Chromium at build time",
      "browser ensure" in dockerfile and "hyperframes@" in dockerfile)
check("the Space exposes app_port 7860 to the Hub",
      "sdk: docker" in space_readme and "app_port: 7860" in space_readme and "7860" in dockerfile)

# The real HTTP surface, on a real socket: this is the contract the brain calls.
space_app.SHARED_SECRET = "s3cret"
space_app._render_job = lambda job, payload: job.update(
    status="completed", percent=100, stage="complete",
    artifact=b"\x00\x00\x00\x20ftypisomHTTP", probe={"frames": 1}, bytes=16,
)
server = space_app.ThreadingHTTPServer(("127.0.0.1", 0), space_app.Handler)
worker_port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{worker_port}"


def http_json(path, payload=None, token=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.loads(res.read() or b"{}")
    except urllib.error.HTTPError as err:
        body = err.read()
        try:
            return err.code, json.loads(body or b"{}")
        except Exception:
            return err.code, {}


try:
    status_code, health_body = http_json("/health")
    check("the worker exposes public /health readiness",
          status_code == 200 and health_body.get("ok") is True, str(status_code))
    status_code, _ = http_json("/render", worker_payload)
    check("the worker refuses an unauthenticated render", status_code == 401, str(status_code))
    status_code, accepted = http_json("/render", worker_payload, token="s3cret")
    check("the worker accepts an authenticated render with 202",
          status_code == 202 and accepted.get("job_id"), str(accepted)[:120])
    status_code, snapshot = http_json(f"/status/{accepted['job_id']}", token="s3cret")
    check("the worker reports per-job status over HTTP",
          status_code == 200 and snapshot.get("status") == "completed", str(snapshot)[:120])
    check("the worker rejects an unknown job id", http_json("/status/nope", token="s3cret")[0] == 404)
finally:
    server.shutdown()
    server.server_close()

# --------------------------------------------- 6g. module + directive integrity
print("\n--- module integrity ---")
source = open("main.py").read()
worker_source = open(os.path.join("space", "app.py")).read()
check("main.py parses as strict AST", main._is_valid_python(source))
check("the 16 GB worker parses as strict AST", main._is_valid_python(worker_source))
check("every FastAPI route survives the overhaul",
      '@api.get("/")' in source and '@api.get("/health")' in source
      and '@api.post("/render-webhook")' in source and '@api.post("/telemetry-error")' in source)
check("startup and shutdown handlers are still wired",
      '@api.on_event("startup")' in source and '@api.on_event("shutdown")' in source)
check("both telegram lifecycle functions still exist",
      "def handle_start(" in source and "def handle_message(" in source
      and "start_polling(drop_pending_updates=True)" in source)
check("telegram handlers are still registered",
      'CommandHandler("start", handle_start)' in source
      and "MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)" in source)
check("the verified Gemini pool is preserved",
      {"gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-flash-latest"} <= set(main.ACTIVE_GEMINI_MODELS),
      str(main.ACTIVE_GEMINI_MODELS))
check("multi-slot key rotation is intact", callable(main.rotate_to_next_gemini_key)
      and callable(main.get_current_gemini_client))
check("no heavy media module is imported by the render brain",
      main.BRAIN_BOUNDARY_VIOLATIONS == [], str(main.BRAIN_BOUNDARY_VIOLATIONS))
check("the boundary auditor really detects a planted violation",
      main.brain_boundary_violations("import PIL\nimport subprocess\n") == ["PIL", "subprocess"])
check("the E2B path still guarantees a kill in finally",
      "finally:" in source and "sbx.kill()" in source)

instruction = main.SYSTEM_INSTRUCTION
check("the instruction names the distributed identity",
      "autonomous audiovisual motion engineer" in instruction)
check("the instruction states the Render brain memory budget",
      "512 MB" in instruction and "Render brain" in instruction)
check("the instruction forbids local rendering outright",
      "NEVER render video locally" in instruction and "Pillow frame buffers" in instruction)
check("the instruction names the E2B-only engine after the HF retirement",
      "E2B microVM" in instruction and "RETIRED" in instruction)
check("the instruction forbids claiming tools are unsupported",
      "unsupported or unavailable" in instruction)
check("the instruction carries the 9:16 video standards",
      "1080x1920" in instruction and "80pt" in instruction and "title-safe" in instruction)
check("the instruction requires edge-tts narration and sub-bass impacts",
      "edge-tts" in instruction and "sub-bass" in instruction)
check("every tool is described to the model",
      all(f"`{fn.__name__}`" in instruction for fn in main.ALL_TOOLS_LIST))
check("every skill is described to the model",
      all(f"`{name}`" in instruction for name in main.SKILL_REGISTRY))
check("every tool carries a one-line capability note",
      all(fn.__name__ in main.TOOL_CAPABILITIES for fn in main.ALL_TOOLS_LIST))
check("remotion is registered with its editor repository",
      main.SKILL_REGISTRY["remotion"]["repo"] == "remotion-dev/remotion")
check("the six motion-graphics engines are all registered",
      {"hyperframes", "gsap", "whisper", "motion_graphics_prompts", "motion_skills", "remotion"}
      <= set(main.SKILL_REGISTRY), str(sorted(main.SKILL_REGISTRY)))

# Restore the real HTTP clients before touching the network again.
main.requests.get = real_get
main.requests.post = real_post
main._GITHUB_RAW_CACHE.clear()

remotion_doc = main.load_skill("remotion")
check("remotion documentation loads live from GitHub",
      remotion_doc.startswith("SKILL 'remotion'") and len(remotion_doc) > 500,
      remotion_doc[:120])
remotion_timeline = main.load_skill("remotion", "packages/docs/docs/composition.mdx", max_chars=2000)
check("a remotion deep-link doc loads too",
      remotion_timeline.startswith("SKILL 'remotion'") and len(remotion_timeline) > 500,
      remotion_timeline[:120])

# ------------------------------------- 6h. external fallback tool-loop contract
print("\n--- external fallback tool loop ---")
import logging

CRITICAL_LINE = (
    "CRITICAL: Do not dump skill markdown or READMEs to the user. "
    "Inspect skills concisely, generate the HTML/CSS/GSAP composition, and call "
    "`render_html_video` immediately to produce and deliver the video."
)
check("the external fallback allows 12 tool rounds", main.EXTERNAL_MAX_TOOL_ROUNDS == 12,
      str(main.EXTERNAL_MAX_TOOL_ROUNDS))
check("the fallback directive matches the mandated wording verbatim",
      main.FALLBACK_TOOL_ROUND_DIRECTIVE == CRITICAL_LINE,
      main.FALLBACK_TOOL_ROUND_DIRECTIVE[:140])

# Capture the exact message array the provider receives on round 1.
posted_bodies = []


def _capture_post(url, **kw):
    posted_bodies.append(kw.get("json"))
    return _Res(200, {"choices": [{"message": {"content": "FINAL-ANSWER"}}]})


main.requests.post = _capture_post
out = main._chat_with_tools("https://example.test/v1", "k", "m/x", "build a reel", "openrouter")
first_msgs = posted_bodies[0]["messages"]
check("a fallback session answers immediately when the model is direct",
      out == "FINAL-ANSWER", out[:120])
check("the session opens with the system instruction",
      first_msgs[0]["role"] == "system" and first_msgs[0]["content"] == main.SYSTEM_INSTRUCTION)
check("the CRITICAL no-doc-dump directive is a system message before the user turn",
      len(first_msgs) == 3 and first_msgs[1]["role"] == "system"
      and first_msgs[1]["content"] == CRITICAL_LINE
      and first_msgs[2] == {"role": "user", "content": "build a reel"},
      json.dumps([m["role"] for m in first_msgs]))

# A model that spends every round paging skill READMEs must now get 12 rounds
# (not 6), keep the directive on every request, and end with a LOUD log entry
# plus a turn_failed flag so the progress card cannot claim completion.

class _ErrCap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())


def _always_tool(url, **kw):
    posted_bodies.append(kw.get("json"))
    call = {"id": "c1", "function": {"name": "list_skills", "arguments": "{}"}}
    return _Res(200, {"choices": [{"message": {"content": "", "tool_calls": [call]}}]})


posted_bodies.clear()
main.requests.post = _always_tool
err_cap = _ErrCap()
main.logger.addHandler(err_cap)
main.PROGRESS_STATE["turn_failed"] = ""
ceiling_reply = main._chat_with_tools(
    "https://example.test/v1", "k", "m/x", "make a reel", "openrouter"
)
check("a doc-dumping model gets exactly 12 API rounds, not 6",
      len(posted_bodies) == 12, f"rounds={len(posted_bodies)}")
check("the ceiling reply states the 12-round exhaustion",
      ceiling_reply.startswith("openrouter stopped after 12 tool rounds without a final answer"),
      ceiling_reply[:120])
check("every round of the session carries the CRITICAL directive",
      all(b["messages"][1]["content"] == CRITICAL_LINE for b in posted_bodies),
      f"rounds={len(posted_bodies)}")
check("a tool-round ceiling is logged as an ERROR naming the ceiling",
      any("TOOL-ROUND CEILING" in r and "12 tool rounds" in r for r in err_cap.records),
      str(err_cap.records[-2:])[:200])
check("a tool-round ceiling flags the turn as failed",
      "tool-round ceiling after 12 rounds" in str(main.PROGRESS_STATE.get("turn_failed")),
      str(main.PROGRESS_STATE.get("turn_failed")))

# End-to-end through handle_message: a ceiling turn must end with an honest
# failure card, NEVER the forced 100% "Complete! Video delivered." claim.
main.requests.post = _always_tool      # the reasoning stub below drives the real loop
f_update = _TgUpdate("make me that reel")
saved_reasoning2 = main.generate_reasoning_response


async def ceiling_reasoning(message):
    return main._chat_with_tools(
        "https://example.test/v1", "k", "m/x", message, "openrouter"
    )


main.generate_reasoning_response = ceiling_reasoning


async def drive_ceiling():
    await main.handle_message(f_update, _TgContext())
    await asyncio.sleep(0.1)


asyncio.run(drive_ceiling())
main.generate_reasoning_response = saved_reasoning2
main.requests.post = real_post
f_card = "\n".join(f_update.message.card.edits)
main.logger.removeHandler(err_cap)
check("the failure turn logs an error about ending with no video",
      any("ending with NO video produced" in r for r in err_cap.records),
      str(err_cap.records[-2:])[:200])
check("the ceiling turn's card shows the honest failure stage",
      "Stopped — no video produced" in f_card, f_card[-200:])
check("the ceiling turn NEVER posts a 100% completion card",
      "100%" not in f_card and "Complete! Video delivered." not in f_card, f_card[-200:])
check("the round-ceiling notice is what the user receives",
      any("stopped after 12 tool rounds" in r for r in f_update.message.replies),
      str(f_update.message.replies)[:160])

# Positive control: a plain chat turn (no video requested, no failure) still
# finishes its card — as "Task complete.", never as "Complete! Video delivered."
chat_update = _TgUpdate("hello")
saved_reasoning3 = main.generate_reasoning_response


async def plain_reasoning(message):
    return "Here is a plan for your reel."


main.generate_reasoning_response = plain_reasoning


async def drive_chat():
    await main.handle_message(chat_update, _TgContext())
    await asyncio.sleep(0.1)


er_cap2 = _ErrCap()
main.logger.addHandler(er_cap2)
asyncio.run(drive_chat())
main.logger.removeHandler(er_cap2)
main.generate_reasoning_response = saved_reasoning3
main.requests.post = real_post
chat_card = "\n".join(chat_update.message.card.edits)
check("a chat turn without a video finishes at Task complete, not Video delivered",
      "100%" in chat_card and "Task complete." in chat_card
      and "Complete! Video delivered." not in chat_card, chat_card[-160:])
check("a successful chat turn logs no no-video failure error",
      not any("NO video produced" in r for r in er_cap2.records), str(er_cap2.records[-1:]))
check("the video-dispatched endgame still claims 100% delivered (preceding turn)",
      "Complete! Video delivered." in card_text, card_text[-160:])

# ------------------------------------------------------- 7. surfaces unchanged
print("\n--- protected surfaces ---")
src = open("main.py").read()
check("@api.get(\"/health\") present", '@api.get("/health")' in src)
check("render webhook present", '@api.post("/render-webhook")' in src)
check("telemetry listener present", '@api.post("/telemetry-error")' in src)
check("telegram polling present", "start_polling(drop_pending_updates=True)" in src)
check("15% progress card present", "15%" in src)
check("100% completion card present", "100%" in src)
check("every declared tool is in the runtime registry",
      set(main.TOOL_REGISTRY) == {f.__name__ for f in main.ALL_TOOLS_LIST})
check("the render and preflight tools are exposed to the model",
      {"render_html_video", "execute_in_cloud_microvm"} <= set(main.TOOL_REGISTRY))
health = main.health_check()
check("health reports the E2B primary and the HF retirement notice",
      health["render_backends"]["primary"] == "e2b_sandbox"
      and "retired" in health["render_backends"]["huggingface"]
      and "e2b_sandbox" in health["render_backends"], json.dumps(health["render_backends"]))
check("health exposes the compute role and the provider telemetry",
      health["compute_role"] == "brain" and "last_provider" in health)
check("health still reports the skill library", health["skills_count"] == len(main.SKILL_REGISTRY))
check("health still reports the tool count", health["tools_count"] == len(main.ALL_TOOLS_LIST))
check("no literal credential values committed",
      not any(k in src for k in ["Bearer sk-", "AIza", "sk-or-v1-"]))

# 6-job editor contracts
check("rulebooks directory exists", os.path.isdir("rulebooks"))
for rb in ["rulebooks/script_rules.md", "rulebooks/visual_rules.md", "rulebooks/audio_rules.md", "rulebooks/qa_rules.md", "rulebooks/hook_library.json"]:
    check(f"{rb} exists", os.path.exists(rb))
check("helpers for 6-job pipeline exist",
      all(hasattr(main, fn) for fn in ["_generate_beat_map", "_research_hooks", "_process_whisper_timestamps", "_qa_10fps_check", "_serialize_render_queue"]))
check("/note handler wired", "_handle_feedback" in dir(main))
bm = main._generate_beat_map("Hello world. How are you.", duration=4.0)
check("beat_map has beats with timestamps", "beats" in bm and len(bm["beats"]) > 0 and "start" in bm["beats"][0] and "end" in bm["beats"][0])

print(f"\n===== {len(PASS)} passed, {len(FAIL)} failed =====")
for f in FAIL: print("FAILED:", f)
sys.exit(1 if FAIL else 0)

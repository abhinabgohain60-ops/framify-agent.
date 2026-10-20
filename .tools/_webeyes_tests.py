# --------------------------------------- 9. web eyes & script ears (agent-reach)
# inspect_web_content reads pages without leaving the brain (no subprocess, no
# media imports); fetch_youtube_transcript runs yt-dlp inside the fake E2B
# sandbox, exactly like the established engine tests do.

check("the web-eyes tools are registered with one-line capability notes",
      "inspect_web_content" in main.TOOL_REGISTRY
      and "fetch_youtube_transcript" in main.TOOL_REGISTRY
      and "inspect_web_content" in main.TOOL_CAPABILITIES
      and "fetch_youtube_transcript" in main.TOOL_CAPABILITIES,
      str(sorted(main.TOOL_CAPABILITIES)))
web_schemas = {s["function"]["name"] for s in main.build_tool_schemas()}
check("the web-eyes tools reach external providers via schema introspection",
      {"inspect_web_content", "fetch_youtube_transcript"} <= web_schemas,
      str(sorted(web_schemas)))
_web_instruction = main.SYSTEM_INSTRUCTION
check("the instruction grants the general web-research role",
      "researcher, web scraper and script analyst" in _web_instruction
      and "`inspect_web_content`" in _web_instruction
      and "`fetch_youtube_transcript`" in _web_instruction)

# --- inspect_web_content ------------------------------------------------------
saved_get = main.requests.get
try:
    check("inspect_web_content rejects non-URLs",
          main.inspect_web_content("not a url").startswith("[FAILED]"))

    class _Res:
        def __init__(self, status_code, text):
            self.status_code, self.text = status_code, text

    def reader_post(url, **kw):
        if url.startswith("https://r.jina.ai/"):
            return _Res(200, "READER TEXT of the article body. " + "Body sentence sits here. " * 14)
        return _Res(0, "")

    main.requests.get = reader_post
    out = main.inspect_web_content("https://example.com/article")
    check("inspect_web_content reads through the agent-reach reader route",
          out.startswith("WEBCONTENT") and "READER TEXT" in out
          and "source: agent-reach reader" in out, out[:200])

    out = main.inspect_web_content("https://example.com/article", max_chars=205)
    check("reader text is truncated exactly at max_chars with a TRUNCATED note",
          len(out.split("\n\n", 1)[1]) == 205 and "TRUNCATED" in out, out[:220])

    def reader_fails(url, **kw):
        if url.startswith("https://r.jina.ai/"):
            raise requests.RequestException("reader down")
        return _Res(200, "<html><body><h1>Direct page</h1><p>Solid text body plus nav.</p></body></html>")

    main.requests.get = reader_fails
    out = main.inspect_web_content("https://example.com/article")
    check("inspect_web_content falls back to a direct fetch and strips HTML",
          "source: direct-fetch" in out and "<html" not in out
          and "Solid text body" in out, out[:200])

    def both_fail(url, **kw):
        raise requests.RequestException("no net")

    main.requests.get = both_fail
    out = main.inspect_web_content("https://example.com/article")
    check("a total fetch failure is reported, not invented",
          out.startswith("[FAILED] could not read"), out[:200])
finally:
    main.requests.get = saved_get

# --- fetch_youtube_transcript ---------------------------------------------------
yt_sbx_cmdHist = []


class _YtRun:
    stdout = ""
    stderr = ""
    exit_code = 0


class _YtSbx:
    """Fake E2B sandbox shaped like the engine fakes: .files.write + .commands.run."""

    def __init__(self, fail_fetch=False):
        self.written = {}
        self.killed = False
        self.fail_fetch = fail_fetch
        self.files = types.SimpleNamespace(write=self.write)
        self.commands = types.SimpleNamespace(run=self._run)

    def write(self, path, data):
        self.written[path] = data

    def _run(self, command, timeout=None):
        yt_sbx_cmdHist.append(command)
        out = _YtRun()
        out.__class__ = _YtRun
        if getattr(self, "fail_fetch", False):
            if "ytdl_fetch.py" in command:
                out.stdout = "ffmpeg noise\nno result here\n"
        elif "ytdl_fetch.py" in command:
            out.stdout = (
                'YT_RESULT {"video_id": "dQw4w9WgXcQ", "title": "Great Talk",'
                ' "duration": 615.0, "uploader": "Example Channel",'
                ' "lang": "en", "chars": 39, "truncated": false, "note": ""}\n'
                "YT_TEXT hello world from the transcript lines"
            )
        return out

    def kill(self):
        self.killed = True
_yt_mod = types.ModuleType("e2b_code_interpreter")


_yt_sbx = _YtSbx()


class _YtSbFactory:
    @staticmethod
    def create(api_key=None, timeout=None):
        return _yt_sbx


_yt_mod.Sandbox = _YtSbFactory
prev_web_mod = sys.modules.get("e2b_code_interpreter")
saved_yt_key = main.E2B_API_KEY
saved_fallback_key = main.E2B_API_KEY

main.E2B_API_KEY = ""
try:
    check("yt-dlp needs the E2B key (it runs in the microVM, not the brain)",
          "E2B_API_KEY is not configured"
          in main.fetch_youtube_transcript("https://www.youtube.com/watch?v=abc"))
finally:
    main.E2B_API_KEY = "TESTKEY"

try:
    check("fetch_youtube_transcript rejects non-URLs",
          main.fetch_youtube_transcript("youtube.com/watch?v=x").startswith("[FAILED]"))
finally:
    sys.modules["e2b_code_interpreter"] = _yt_mod
    main.E2B_API_KEY = "TESTKEY"

try:
    rep = main.fetch_youtube_transcript(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ", langs="en,en-US"
    )
    check("the transcript fetch pipes the task through a JSON file (no shell injection)",
          any("ytdl_task.json" in w for w in _yt_sbx.written), str(list(_yt_sbx.written)))
    check("the sandbox installs and runs yt-dlp",
          any("pip install" in c and "yt-dlp" in c for c in yt_sbx_cmdHist), str(yt_sbx_cmdHist[:2]))
    check("the transcript report carries title/duration/uploader metadata",
          "TRANSCRIPT dQw4w9WgXcQ" in rep and "title: Great Talk" in rep
          and "duration: 615.0s" in rep and "uploader: Example Channel" in rep,
          rep[:300])
    check("the transcript text is returned as plain prose",
          "hello world from the transcript lines" in rep)
    check("the yt sandbox is always killed", _yt_sbx.killed)
finally:
    sys.modules["e2b_code_interpreter"] = prev_web_mod
    main.E2B_API_KEY = saved_yt_key

# Failure path: no YT_RESULT/YT_ERROR in the sandbox output -> loud [FAILED] tail,
# never a fake transcript.
_yt_fail_sbx = _YtSbx(fail_fetch=True)
_yt_fail_mod = types.ModuleType("e2b_code_interpreter")


class _YtFailSbFactory:
    @staticmethod
    def create(api_key=None, timeout=None):
        return _yt_fail_sbx


_yt_fail_mod.Sandbox = _YtFailSbFactory
main.E2B_API_KEY = "TESTKEY"
try:
    sys.modules["e2b_code_interpreter"] = _yt_fail_mod
    rep = main.fetch_youtube_transcript("https://www.youtube.com/watch?v=zQw4w9WgXcQ")
    check("an unreadable yt outcome is a loud [FAILED], never a fake transcript",
          rep.startswith("[FAILED]"), rep[:220])
finally:
    sys.modules["e2b_code_interpreter"] = prev_web_mod
    main.E2B_API_KEY = saved_yt_key

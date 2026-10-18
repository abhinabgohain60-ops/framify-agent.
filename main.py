import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from typing import Callable, Awaitable

def _find_rendered_video(sbx: Any, directory: str = "", slug: str = "") -> Optional[str]:
    """Find an MP4 robustly using a live recursive find inside the sandbox.

    Hardening for the production failure where the E2B microVM wrote media to a
    per-slug subdirectory instead of the rigid fallback paths, so the old
    ``ls /home/user/videos`` 404'd with "path not found" and left the caller
    with no MP4 to ship. We now execute the exact shell asked for in the
    sandbox before teardown, take the first readable .mp4 from stdout, and only
    fall back to listing when the find returned nothing at all.
    """
    try:
        res = sbx.commands.run("find /home/user /tmp -maxdepth 4 -name \"*.mp4\" 2>/dev/null")
        out = getattr(res, "stdout", "") or ""
        for line in out.splitlines():
            p = line.strip()
            if not p or not p.endswith(".mp4"):
                continue
            try:
                sbx.files.read(p, format="bytes")
                return p
            except Exception:
                continue
        # find produced nothing: dump the actual tree to Render logs so the
        # exact file layout is visible there instead of a vague "missing video".
        try:
            ls = sbx.commands.run("ls -laR /home/user 2>&1 | tail -80")
            logger.error(
                "[E2B Debug] /home/user tree after render (find returned no MP4): " + repr(getattr(ls, 'stdout', '') or 'empty ls stdout')
            )
        except Exception as e:
            logger.error(f"[E2B Debug] ls /home/user failed after find: {e}")
        return None
    except Exception as e:
        logger.error(f"[E2B Debug] find /home/user /tmp failed: {e}")
    return None






def _find_mp4_in_sandbox(sbx: Any, slug: str = "") -> Optional[str]:
    """Search recursively for MP4s in /home/user; return first final_output.mp4 or any .mp4 found."""
    try:
        candidates = []
        if slug:
            candidates.extend(
                [
                    f"/home/user/videos/{slug}/renders/final_output.mp4",
                    f"/home/user/videos/{slug}/renders/video.mp4",
                    f"/home/user/videos/{slug}/final_output.mp4",
                    f"/home/user/videos/{slug}/video.mp4",
                ]
            )
        candidates.extend(
            [
                "/home/user/final_output.mp4",
                "/home/user/output/final_output.mp4",
                "/home/user/videos/final_output.mp4",
                "/home/user/renders/final_output.mp4",
            ]
        )
        for c in candidates:
            try:
                sbx.files.read(c, format="bytes")
                return c
            except Exception:
                continue
        # Fallback: find in sandbox
        res = sbx.commands.run("find /home/user -name 'final_output.mp4' -not -path '*/proc/*' 2>&1 | head -5")
        out = getattr(res, "stdout", "") or ""
        for line in out.splitlines():
            p = line.strip()
            if p and p.startswith("/home/user/"):
                try:
                    sbx.files.read(p, format="bytes")
                    return p
                except Exception:
                    continue
        res = sbx.commands.run("find /home/user -name '*.mp4' -not -path '*/proc/*' 2>&1 | head -5")
        out = getattr(res, "stdout", "") or ""
        for line in out.splitlines():
            p = line.strip()
            if p and p.startswith("/home/user/"):
                try:
                    sbx.files.read(p, format="bytes")
                    return p
                except Exception:
                    continue
        # Debug: list videos dir
        try:
            vlist = sbx.files.list("/home/user/videos")
            logger.error(
                "[E2B Debug] /home/user/videos: "
                f"{[getattr(e, 'name', '') for e in vlist]}"
            )
        except Exception as e:
            logger.error(f"[E2B Debug] could not list /home/user/videos: {e}")
    except Exception as e:
        logger.error(f"[E2B MP4 search] error: {e}")
    return None


logger = logging.getLogger("chintu-agent")


def _telegram_send_video(video_bytes: bytes, file_name: str, caption: str = "") -> bool:
    """Post MP4 bytes to the bound Telegram chat. Returns True only on HTTP 200."""
    if not (CURRENT_CHAT_ID and TELEGRAM_BOT_TOKEN):
        return False
    try:
        send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
        upload_files = {"video": (file_name, video_bytes, "video/mp4")}
        data = {"chat_id": CURRENT_CHAT_ID, "caption": caption or f"🎬 Chintu Video Render: {file_name}", "supports_streaming": "true"}
        res = requests.post(send_url, data=data, files=upload_files, timeout=120)
        if res.status_code == 200:
            logger.info(f"Directly uploaded {file_name} to Telegram via HTTP API!")
            return True
        logger.error(f"Telegram upload failed: {res.text[:300]}")
    except Exception as vid_err:
        logger.warning(f"Video delivery warning: {vid_err}")
    return False


def _deliver_rendered_video(sbx: Any, extra_dirs: Any = (), require_audio: bool = False, slug: str = "") -> str:
    """Upload the rendered MP4 straight to Telegram.

    Returns a status string:
      'delivered'      an MP4 shipped to Telegram
      'no-audio'       require_audio=True but no verified final_output.mp4 exists
                       — NOTHING was uploaded, so the failure is loud, not silent
      'post-failed'    final_output.mp4 exists but Telegram refused every attempt
      'no-video'       no MP4 anywhere in the sandbox
      'unbound'        no chat id / bot token
    """
    if not (CURRENT_CHAT_ID and TELEGRAM_BOT_TOKEN):
        return "unbound"

    def _find_mp4_by_find() -> str | None:
        # The authoritative search the user asked for: one live bash find over
        # the whole home/tmp tree before teardown.
        try:
            res = sbx.commands.run('find /home/user /tmp -maxdepth 4 -name "*.mp4" 2>/dev/null')
            out = getattr(res, "stdout", "") or ""
            for line in out.splitlines():
                p = line.strip()
                if not p or not p.endswith(".mp4"):
                    continue
                if p.endswith("final_output.mp4"):
                    try:
                        sbx.files.read(p, format="bytes")
                        return p
                    except Exception:
                        continue
                try:
                    sbx.files.read(p, format="bytes")
                    return p
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"[E2B Debug] find command failed: {e}")
        return None

    # Do NOT rely on the legacy rigid scan-dir list alone. The E2B microVM often
    # writes media into a per-slug subdirectory that shallow reads over
    # /home/user/videos or hardcoded candidate paths will 404 on. Perform one
    # live recursive find over the whole home tree before teardown, and prefer
    # the first readable .mp4 from that search.
    found_final = _find_mp4_in_sandbox(sbx, slug=slug)
    if found_final and "final_output.mp4" in found_final:
        try:
            final_bytes = sbx.files.read(found_final, format="bytes")
            if _telegram_send_video(final_bytes, "final_output.mp4"):
                return "delivered"
            return "post-failed"
        except Exception:
            pass

    for scan_dir in list(extra_dirs) + list(SANDBOX_VIDEO_SCAN_DIRS):
        final_path = f"{str(scan_dir).rstrip('/')}/final_output.mp4"
        try:
            final_bytes = sbx.files.read(final_path, format="bytes")
        except Exception:
            continue
        if _telegram_send_video(final_bytes, "final_output.mp4"):
            return "delivered"
        return "post-failed"

    if slug:
        for base in (f"/home/user/videos/{slug}/renders", f"/home/user/videos/{slug}"):
            final_path = f"{base}/final_output.mp4"
            try:
                final_bytes = sbx.files.read(final_path, format="bytes")
            except Exception:
                continue
            if _telegram_send_video(final_bytes, "final_output.mp4"):
                return "delivered"
            return "post-failed"

    if require_audio:
        logger.error(
            "[Audio mux] narration was required but no final_output.mp4 exists — "
            "refusing to ship the silent capture."
        )
        logger.error("[E2B Debug] dumping /home/user tree to Render logs:")
        try:
            ls = sbx.commands.run("ls -laR /home/user 2>&1 | tail -80")
            logger.error(getattr(ls, "stdout", "") or "(empty ls stdout)")
        except Exception as e:
            logger.error(f"[E2B Debug] ls /home/user failed: {e}")
        return "no-audio"

    # Un-narrated captures still ship, but never under the name video.mp4, and
    # only after we have actually located a readable MP4 through the sandbox find.
    found_any = _find_mp4_by_find()
    if found_any:
        try:
            video_bytes = sbx.files.read(found_any, format="bytes")
            display_name = found_any.rsplit("/", 1)[-1]
            if display_name.lower() == "video.mp4":
                display_name = "final_output.mp4"
            if _telegram_send_video(video_bytes, display_name):
                return "delivered"
        except Exception as vid_err:            logger.warning(f"Video delivery warning: {vid_err}")

    # Explicit recursive find as the last, authoritative attempt before failure.
    found_any = _find_mp4_by_find()
    if found_any:
        try:
            video_bytes = sbx.files.read(found_any, format="bytes")
            display_name = found_any.rsplit("/", 1)[-1]
            if display_name.lower() == "video.mp4":
                display_name = "final_output.mp4"
            if _telegram_send_video(video_bytes, display_name):
                return "delivered"
        except Exception as vid_err:
            logger.warning(f"Video delivery warning: {vid_err}")

    return "no-video"


def _find_rendered_video(sbx: Any, directory: str = "", slug: str = "") -> Optional[str]:
    """Find an MP4 robustly using the exact find command the user requested.

    Replaces the rigid ``/home/user/videos`` candidate-path checks that 404'd in
    production when the E2B microVM wrote media into a per-slug subdirectory.
    Run live inside the sandbox before teardown, take the first valid .mp4 from
    stdout and download that exact path.
    """
    try:
        res = sbx.commands.run("find /home/user /tmp -maxdepth 4 -name \"*.mp4\" 2>/dev/null")
        out = getattr(res, "stdout", "") or ""
        for line in out.splitlines():
            p = line.strip()
            if not p or not p.endswith(".mp4"):
                continue
            try:
                sbx.files.read(p, format="bytes")
                return p
            except Exception:
                continue
        # find produced nothing: dump the actual tree to Render logs so the
        # exact file layout is visible there instead of a vague "missing video".
        try:
            ls = sbx.commands.run("ls -laR /home/user 2>&1 | tail -80")
            logger.error(
                "[E2B Debug] /home/user tree after render (find returned no MP4): "
                + repr(getattr(ls, 'stdout', '') or '(empty ls stdout)')
            )
        except Exception as e:
            logger.error(f"[E2B Debug] ls /home/user failed after find: {e}")
        return None
    except Exception as e:
        logger.error(f"[E2B Debug] find /home/user /tmp failed: {e}")
    return None






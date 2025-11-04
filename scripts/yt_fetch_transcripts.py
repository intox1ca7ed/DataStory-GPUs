#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "data-raw"
TRANSCRIPTS_DIR = RAW_DATA_DIR / "transcripts"
TMP_DIR = RAW_DATA_DIR / "tmp"
LOGS_DIR = RAW_DATA_DIR / "logs"
MANIFEST_PATH = RAW_DATA_DIR / "manifest.csv"
INDEX_PATH = RAW_DATA_DIR / "index.parquet"

DEFAULT_SUB_LANGS = ["en", "en-US", "en-GB"]
RATE_LIMIT_PATTERNS = ["http error 429", "too many requests"]
MIN_SLEEP_BETWEEN_VIDEOS = 6
MAX_SLEEP_BETWEEN_VIDEOS = 10

README_SNIPPET = """
Prerequisites:
  pip install yt-dlp pandas pyarrow

Usage:
  python scripts/yt_fetch_transcripts.py
  python scripts/yt_fetch_transcripts.py --max-videos 5
  python scripts/yt_fetch_transcripts.py --sub-langs en,en-US --include-all-subs
  python scripts/yt_fetch_transcripts.py --smoke-test

Behavior:
  Phase 2 script that fetches subtitles/transcripts only, reading ./data-raw/manifest.csv produced by yt_fetch_comments.py.
  Outputs land in ./data-raw/transcripts/ and the manifest/index files are updated in place.
  Logs rotate at ./data-raw/logs/transcripts.log (max 10MB, 5 backups). Tail via: tail -f data-raw/logs/transcripts.log or Get-Content ... -Wait on PowerShell.
  By default only English subtitle variants (en, en-US, en-GB) are requested to reduce rate-limit risk; use --sub-langs/--include-all-subs to override.
  A 5-8 second cooldown runs between videos to avoid HTTP 429 responses, and retries/backoff are applied if YouTube throttles requests.
""".strip()


@dataclass
class PathConfig:
    transcripts_dir: Path
    tmp_dir: Path
    logs_dir: Path
    manifest_path: Path
    index_path: Path


@dataclass
class TranscriptResult:
    video_id: str
    status_transcripts: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube transcripts/subtitles and update manifest (phase 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=README_SNIPPET,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Path to manifest CSV produced by yt_fetch_comments.py (default: {MANIFEST_PATH})",
    )
    parser.add_argument("--max-videos", type=int, help="Only process first N manifest rows needing transcripts.")
    parser.add_argument("--force", action="store_true", help="Re-download transcripts even if files already exist.")
    parser.add_argument(
        "--sub-langs",
        type=str,
        default=",".join(DEFAULT_SUB_LANGS),
        help="Comma-separated subtitle language codes to request (default: en,en-US,en-GB).",
    )
    parser.add_argument(
        "--include-all-subs",
        action="store_true",
        help="Append 'all' to the subtitle language list to request every track (may trigger more 429s).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Limit to <=5 videos and verify transcript outputs after processing.",
    )
    return parser.parse_args()


def ensure_directories() -> PathConfig:
    for path in (TRANSCRIPTS_DIR, TMP_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return PathConfig(TRANSCRIPTS_DIR, TMP_DIR, LOGS_DIR, MANIFEST_PATH, INDEX_PATH)


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("yt_fetch_transcripts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty; run yt_fetch_comments.py first.")
    return rows


def parse_int(value: Optional[str]) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return 0


def select_manifest_rows(rows: List[Dict[str, str]], force: bool) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for row in rows:
        status = row.get("status_transcripts", "")
        existing = parse_int(row.get("num_transcript_files"))
        if force or status != "ok" or existing == 0:
            if row.get("video_id") and row.get("url"):
                selected.append(row)
    return selected


def build_sub_langs(arg_string: str, include_all: bool) -> str:
    seen = set()
    langs: List[str] = []
    for lang in arg_string.split(","):
        lang = lang.strip()
        if not lang:
            continue
        if lang not in seen:
            seen.add(lang)
            langs.append(lang)
    if include_all and "all" not in seen:
        langs.append("all")
    return ",".join(langs) if langs else "all"


def is_rate_limited(stderr_text: str) -> bool:
    lowered = (stderr_text or "").lower()
    return any(pattern in lowered for pattern in RATE_LIMIT_PATTERNS)


def invoke_yt_dlp_transcripts(
    video_id: str,
    url: str,
    tmp_run_dir: Path,
    lang_arg: str,
    logger: logging.Logger,
    attempts: int = 3,
) -> Tuple[bool, str, bool]:
    langs = [s.strip() for s in lang_arg.split(",") if s.strip()]
    base_cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-format",
        "vtt",
        "--output",
        "%(id)s/%(id)s",
        url,
    ]
    if any(l.lower() == "all" for l in langs):
        base_cmd.insert(5, "--all-subs")
    else:
        base_cmd += ["--sub-langs", lang_arg]
    backoffs = [5, 15, 30]
    tmp_run_dir.mkdir(parents=True, exist_ok=True)
    last_stderr = ""
    was_rate_limited = False
    for attempt in range(1, attempts + 1):
        logger.info("yt-dlp transcripts attempt %s for %s", attempt, video_id)
        proc = subprocess.run(
            base_cmd,
            capture_output=True,
            text=True,
            cwd=tmp_run_dir,
            check=False,
        )
        last_stderr = proc.stderr or ""
        if proc.returncode == 0:
            logger.info("yt-dlp transcripts succeeded for %s on attempt %s", video_id, attempt)
            return True, last_stderr, False
        if is_rate_limited(last_stderr):
            was_rate_limited = True
            logger.warning("yt-dlp hit rate limit for %s (attempt %s)", video_id, attempt)
        else:
            logger.warning("yt-dlp transcripts failed for %s (attempt %s rc=%s)", video_id, attempt, proc.returncode)
            logger.warning(last_stderr.strip() or "yt-dlp stderr empty")
        if attempt == attempts:
            fallback_cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-auto-subs",
                "--sub-format",
                "vtt",
                "--output",
                "%(id)s/%(id)s",
                url,
            ]
            logger.info("Final fallback: auto-subs only for %s", video_id)
            proc = subprocess.run(
                fallback_cmd,
                capture_output=True,
                text=True,
                cwd=tmp_run_dir,
                check=False,
            )
            last_stderr = proc.stderr or ""
            if proc.returncode == 0:
                logger.info("Fallback succeeded for %s", video_id)
                return True, last_stderr, is_rate_limited(last_stderr)
        if attempt < attempts:
            sleep_for = backoffs[min(attempt - 1, len(backoffs) - 1)]
            logger.info("Sleeping %ss before retrying %s", sleep_for, video_id)
            time.sleep(sleep_for)
    logger.warning("yt-dlp transcripts exhausted retries for %s", video_id)
    return False, last_stderr, was_rate_limited


def move_transcripts(video_id: str, tmp_video_dir: Path, transcripts_dir: Path) -> List[Path]:
    download_subdir = tmp_video_dir / video_id
    moved: List[Path] = []
    if download_subdir.exists():
        for transcript_file in download_subdir.glob("*.vtt"):
            stem = transcript_file.stem
            lang = "und"
            prefix = f"{video_id}."
            if stem.startswith(prefix):
                lang_candidate = stem[len(prefix) :]
                if lang_candidate:
                    lang = lang_candidate
            else:
                parts = stem.split(".")
                if len(parts) > 1:
                    lang = parts[-1]
            safe_lang = re.sub(r"[^A-Za-z0-9_-]+", "-", lang) or "und"
            dest = transcripts_dir / f"{video_id}.{safe_lang}.vtt"
            shutil.move(str(transcript_file), dest)
            moved.append(dest)
    if download_subdir.exists():
        shutil.rmtree(download_subdir, ignore_errors=True)
    if tmp_video_dir.exists():
        shutil.rmtree(tmp_video_dir, ignore_errors=True)
    return moved


def write_manifest(manifest_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "video_id",
            "url",
            "gpu_model",
            "brand",
            "video_title",
            "status_comments",
            "status_transcripts",
            "comments_path",
            "transcripts_dir",
            "info_path",
            "error_message",
            "started_at",
            "finished_at",
            "elapsed_sec",
            "num_transcript_files",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rebuild_index(manifest_rows: List[Dict[str, str]], index_path: Path) -> None:
    records: List[Dict[str, object]] = []
    for row in manifest_rows:
        records.append(
            {
                "video_id": row.get("video_id", ""),
                "url": row.get("url", ""),
                "gpu_model": row.get("gpu_model", ""),
                "brand": row.get("brand", ""),
                "video_title": row.get("video_title", ""),
                "comments_json_exists": Path(row.get("comments_path", "")).exists(),
                "info_json_exists": Path(row.get("info_path", "")).exists(),
                "transcript_file_count": parse_int(row.get("num_transcript_files")),
            }
        )
    df = pd.DataFrame(records)
    df.to_parquet(index_path, index=False)


def summarize(results: List[TranscriptResult]) -> Dict[str, int]:
    counts = {
        "total": len(results),
        "transcripts_ok": 0,
        "transcripts_none": 0,
        "transcripts_error": 0,
    }
    for item in results:
        if item.status_transcripts == "ok":
            counts["transcripts_ok"] += 1
        elif item.status_transcripts == "none":
            counts["transcripts_none"] += 1
        else:
            counts["transcripts_error"] += 1
    return counts


def verify_smoke_test(paths: PathConfig, rows: List[Dict[str, str]], logger: logging.Logger) -> bool:
    checks: List[str] = []
    if not paths.transcripts_dir.exists():
        checks.append(f"Missing transcripts directory: {paths.transcripts_dir}")
    if not any(paths.transcripts_dir.glob("*.vtt")):
        checks.append("No transcript files were created.")
    successful_rows = [
        row for row in rows if row.get("status_transcripts") in {"ok", "none"}
    ]
    if not successful_rows:
        checks.append("Manifest lacks transcript status rows (ok/none).")
    if checks:
        logger.error("SMOKE TEST FAIL")
        for issue in checks:
            logger.error(" - %s", issue)
        print("SMOKE TEST FAIL")
        for issue in checks:
            print(f" - {issue}")
        return False
    logger.info("SMOKE TEST PASS")
    print("SMOKE TEST PASS")
    return True


def main() -> int:
    args = parse_args()
    if args.smoke_test and (args.max_videos is None or args.max_videos > 5):
        args.max_videos = 5
    paths = ensure_directories()
    logger = configure_logging(paths.logs_dir / "transcripts.log")
    try:
        manifest_rows = read_manifest(args.manifest)
    except Exception as exc:
        logger.error(str(exc))
        print(str(exc), file=sys.stderr)
        return 1
    candidates = select_manifest_rows(manifest_rows, args.force)
    if not candidates:
        logger.info("No manifest rows need transcripts (use --force to re-download).")
        return 0
    if args.max_videos:
        candidates = candidates[: args.max_videos]
    lang_arg = build_sub_langs(args.sub_langs, args.include_all_subs)
    results: List[TranscriptResult] = []
    for idx, row in enumerate(candidates, start=1):
        video_id = row["video_id"]
        url = row["url"]
        logger.info("START transcripts %s (%s)", video_id, row.get("video_title", ""))
        tmp_run_dir = paths.tmp_dir / f"subs_{video_id}"
        if tmp_run_dir.exists():
            shutil.rmtree(tmp_run_dir, ignore_errors=True)
        existing = list(TRANSCRIPTS_DIR.glob(f"{video_id}.*.vtt"))
        skip = not args.force and existing
        started_at = datetime.now(timezone.utc)
        error_message = row.get("error_message", "")
        status_transcripts = row.get("status_transcripts", "none")
        if skip:
            logger.info("Skipping transcripts for %s (already present)", video_id)
            transcript_files = existing
            stderr_text = ""
            success = True
        else:
            success, stderr_text, rate_limited = invoke_yt_dlp_transcripts(
                video_id, url, tmp_run_dir, lang_arg, logger
            )
            transcript_files = move_transcripts(video_id, tmp_run_dir, paths.transcripts_dir) if success else []
            if not success:
                error_snippet = (stderr_text or "").strip()[:1000]
                if error_snippet:
                    if error_message:
                        error_message = f"{error_message} | transcripts: {error_snippet}"
                    else:
                        error_message = f"transcripts: {error_snippet}"
        if success:
            if transcript_files:
                status_transcripts = "ok"
            else:
                status_transcripts = "none"
        else:
            status_transcripts = "error"
        row["status_transcripts"] = status_transcripts
        row["num_transcript_files"] = str(len(transcript_files) if not skip else len(existing))
        row["transcripts_dir"] = str(paths.transcripts_dir)
        row["error_message"] = error_message
        results.append(TranscriptResult(video_id, status_transcripts))
        logger.info(
            "END transcripts %s status=%s duration=%.2fs",
            video_id,
            status_transcripts,
            (datetime.now(timezone.utc) - started_at).total_seconds(),
        )
        if idx < len(candidates):
            time.sleep(random.uniform(MIN_SLEEP_BETWEEN_VIDEOS, MAX_SLEEP_BETWEEN_VIDEOS))
            if idx % 10 == 0:
                extra_sleep = random.uniform(60, 120)
                logger.info("Extended cooldown %.1fs after %s videos", extra_sleep, idx)
                time.sleep(extra_sleep)
    write_manifest(paths.manifest_path, manifest_rows)
    rebuild_index(manifest_rows, paths.index_path)
    summary = summarize(results)
    logger.info("Summary: %s", ", ".join(f"{k}={v}" for k, v in summary.items()))
    if args.smoke_test:
        ok = verify_smoke_test(paths, manifest_rows, logger)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

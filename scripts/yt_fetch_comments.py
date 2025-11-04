#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
COMMENTS_DIR = RAW_DATA_DIR / "comments"
INFO_DIR = RAW_DATA_DIR / "info"
TRANSCRIPTS_DIR = RAW_DATA_DIR / "transcripts"
TMP_DIR = RAW_DATA_DIR / "tmp"
LOGS_DIR = RAW_DATA_DIR / "logs"
MANIFEST_PATH = RAW_DATA_DIR / "manifest.csv"
INDEX_PATH = RAW_DATA_DIR / "index.parquet"

DISABLED_PATTERNS = [
    "comments are disabled",
    "this video has disabled comments",
    "comments have been disabled",
    "disabled for this video",
    "ѧ��ѬѬ��ѫ�'���?���� ��'ѧѯ�z����ѫ�<",
    "comentarios desactivados",
    "kommentare deaktiviert",
]

MANIFEST_FIELDS = [
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
]

README_SNIPPET = """
Prerequisites:
  pip install yt-dlp pandas pyarrow

Usage:
  python scripts/yt_fetch_comments.py
  python scripts/yt_fetch_comments.py --max-videos 5
  python scripts/yt_fetch_comments.py --yt-extractor-args "youtube:max_comments=0,comment_sort=top"
  python scripts/yt_fetch_comments.py --smoke-test

Flags:
  --yt-extractor-args lets power users tweak yt-dlp comment behavior (e.g., max_comments=0, comment_sort=top/recent).
  If yt-dlp rejects the provided extractor args, the script automatically retries without them.
  --smoke-test caps processing to 5 videos (unless a smaller --max-videos is supplied) and validates directory layout.
  --force re-downloads even when files already exist; --csv selects an alternate CSV path.

Behavior:
  Phase 1 script that fetches comments + info JSON only; transcripts are handled separately.
  Outputs live under ./data-raw/ (comments/, info/, transcripts/, logs/, tmp/).
  Logs rotate at ./data-raw/logs/comments.log (max 10MB, 5 backups). Tail via: tail -f data-raw/logs/comments.log  (Linux/macOS) or Get-Content data-raw/logs/comments.log -Wait (PowerShell).
  Comments statuses distinguish disabled vs none using localized patterns.
  Manifest is written to ./data-raw/manifest.csv and will later be updated by the transcript fetcher.
""".strip()


def resolve_default_csv() -> Path:
    candidates = [
        Path("/mnt/data/youtube_videos.csv"),
        REPO_ROOT / "youtube_videos.csv",
        Path.cwd() / "youtube_videos.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


BASE_CSV = resolve_default_csv()


@dataclass
class PathConfig:
    comments_dir: Path
    info_dir: Path
    transcripts_dir: Path
    tmp_dir: Path
    logs_dir: Path
    manifest_path: Path
    index_path: Path


@dataclass
class ProcessResult:
    manifest_row: Dict[str, object]
    index_row: Dict[str, object]
    status_comments: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube comments and metadata (phase 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=README_SNIPPET,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=BASE_CSV,
        help=f"Path to CSV with Youtube_URL data (default: {BASE_CSV})",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if artifacts already exist.")
    parser.add_argument("--max-videos", type=int, help="Only process the first N videos for dry runs.")
    parser.add_argument(
        "--yt-extractor-args",
        type=str,
        help="Raw value to pass as --extractor-args to yt-dlp (use quotes).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Limit to <=5 videos and validate outputs after processing.",
    )
    return parser.parse_args()


def ensure_directories() -> PathConfig:
    for path in (COMMENTS_DIR, INFO_DIR, TRANSCRIPTS_DIR, TMP_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return PathConfig(COMMENTS_DIR, INFO_DIR, TRANSCRIPTS_DIR, TMP_DIR, LOGS_DIR, MANIFEST_PATH, INDEX_PATH)


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("yt_fetch_comments")
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


def read_catalog(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV input not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"Youtube_URL", "Video_Title"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"CSV missing required columns: {missing_str}")
    return df


VIDEO_REGEXES = [
    r"(?:v=|/videos/|embed/|shorts/|v/)(?P<id>[A-Za-z0-9_-]{11})",
    r"youtu\.be/(?P<id>[A-Za-z0-9_-]{11})",
    r"youtube\.com/playlist\?list=(?P<id>[A-Za-z0-9_-]{11})",
]


def extract_video_id(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    cleaned = url.strip()
    for pattern in VIDEO_REGEXES:
        match = re.search(pattern, cleaned)
        if match:
            candidate = match.group("id")
            if len(candidate) == 11:
                return candidate
    if len(cleaned) == 11 and all(ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for ch in cleaned):
        return cleaned
    return None


def invoke_yt_dlp_comments(
    video_id: str,
    url: str,
    tmp_run_dir: Path,
    extractor_args: Optional[str],
    logger: logging.Logger,
) -> Tuple[bool, int, str, int]:
    base_cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-info-json",
        "--write-comments",
        "--output",
        "%(id)s/%(id)s",
        url,
    ]
    attempts = 3
    backoffs = [2, 5]
    last_stderr = ""
    tmp_run_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        cmd = base_cmd[:-1]
        drop_extractor = extractor_args and attempt == attempts
        if extractor_args and not drop_extractor:
            cmd += ["--extractor-args", extractor_args]
        cmd.append(url)
        logger.info("yt-dlp attempt %s for %s (drop_extractor=%s)", attempt, video_id, bool(drop_extractor))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=tmp_run_dir,
            check=False,
        )
        last_stderr = proc.stderr or ""
        if proc.returncode == 0:
            logger.info("yt-dlp succeeded on attempt %s for %s", attempt, video_id)
            return True, attempt, last_stderr, proc.returncode
        logger.warning("yt-dlp failed on attempt %s for %s (rc=%s)", attempt, video_id, proc.returncode)
        logger.warning(last_stderr.strip() or "yt-dlp stderr was empty")
        if attempt < attempts:
            time.sleep(backoffs[min(attempt - 1, len(backoffs) - 1)])
    return False, attempts, last_stderr, proc.returncode


def detect_disabled(stderr_text: str) -> bool:
    lowered = (stderr_text or "").lower()
    return any(pattern.lower() in lowered for pattern in DISABLED_PATTERNS)


def count_comments(comments_path: Path) -> Optional[int]:
    if not comments_path.exists():
        return None
    try:
        text = comments_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not text:
        return 0
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        lowered = text.lower()
        return lowered.count('"comment"')
    counts: List[int] = []
    if isinstance(data, list):
        counts.append(len(data))
    elif isinstance(data, dict):
        for key in ("comments", "entries", "items", "commentEntries"):
            value = data.get(key)
            if isinstance(value, list):
                counts.append(len(value))
        if "contents" in data and isinstance(data["contents"], list):
            counts.append(len(data["contents"]))
    if counts:
        return max(counts)
    if isinstance(data, dict):
        lowered = json.dumps(data).lower()
        return lowered.count('"comment"')
    return 0


def ensure_comments_from_info(info_path: Path, comments_dest: Path, logger: logging.Logger) -> Optional[Path]:
    if not info_path.exists():
        return None
    try:
        info_data = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse info JSON for %s: %s", info_path, exc)
        return None
    comments = info_data.get("comments")
    if not comments:
        return None
    try:
        comments_dest.parent.mkdir(parents=True, exist_ok=True)
        comments_dest.write_text(json.dumps(comments, ensure_ascii=False), encoding="utf-8")
        return comments_dest
    except Exception as exc:
        logger.warning("Unable to write extracted comments for %s: %s", info_path, exc)
        return None


def move_comment_artifacts(
    video_id: str,
    tmp_video_dir: Path,
    comments_dir: Path,
    info_dir: Path,
    logger: logging.Logger,
) -> Tuple[Optional[Path], Optional[Path]]:
    download_subdir = tmp_video_dir / video_id
    comments_src = None
    info_src = None
    if download_subdir.exists():
        comments_candidates = sorted(download_subdir.glob("*.comments.json"))
        info_candidates = sorted(download_subdir.glob("*.info.json"))
        comments_src = comments_candidates[0] if comments_candidates else None
        info_src = info_candidates[0] if info_candidates else None
    comments_dest = comments_dir / f"{video_id}.comments.json"
    info_dest = info_dir / f"{video_id}.info.json"
    if comments_src:
        shutil.move(str(comments_src), comments_dest)
    if info_src:
        shutil.move(str(info_src), info_dest)
    if not comments_src and info_dest.exists():
        extracted = ensure_comments_from_info(info_dest, comments_dest, logger)
        if extracted:
            comments_src = extracted
    if download_subdir.exists():
        shutil.rmtree(download_subdir, ignore_errors=True)
    if tmp_video_dir.exists():
        shutil.rmtree(tmp_video_dir, ignore_errors=True)
    return (comments_dest if comments_src else None, info_dest if info_src else None)


def gather_existing_transcripts(video_id: str) -> List[Path]:
    return list(TRANSCRIPTS_DIR.glob(f"{video_id}.*.vtt"))


def summarize_status_counts(results: List[ProcessResult]) -> Dict[str, int]:
    counts = {
        "total": len(results),
        "comments_ok": 0,
        "comments_disabled": 0,
        "comments_none": 0,
        "comments_error": 0,
    }
    for result in results:
        sc = result.status_comments
        if sc == "ok":
            counts["comments_ok"] += 1
        elif sc == "disabled":
            counts["comments_disabled"] += 1
        elif sc == "none":
            counts["comments_none"] += 1
        else:
            counts["comments_error"] += 1
    return counts


def determine_comment_status(
    return_code: int,
    comments_path: Optional[Path],
    stderr_text: str,
) -> Tuple[str, Optional[int]]:
    disabled = detect_disabled(stderr_text)
    if return_code != 0:
        return ("disabled" if disabled else "error"), None
    if comments_path and comments_path.exists():
        count = count_comments(comments_path)
        if count == 0:
            return "none", count
        return "ok", count
    if disabled:
        return "disabled", None
    return "none", 0


def process_row(
    row: Dict[str, object],
    args: argparse.Namespace,
    paths: PathConfig,
    logger: logging.Logger,
) -> ProcessResult:
    url = str(row.get("Youtube_URL", "")).strip()
    video_title = str(row.get("Video_Title", "")).strip()
    brand = str(row.get("Brand", "")).strip()
    gpu_model = str(row.get("GPU_Model", "")).strip()
    video_id = extract_video_id(url)
    started_at = datetime.now(timezone.utc)
    error_message = ""
    status_comments = "error"
    if not video_id:
        error_message = "Unable to parse video ID"
        finished_at = datetime.now(timezone.utc)
        manifest_row = {
            "video_id": "",
            "url": url,
            "gpu_model": gpu_model,
            "brand": brand,
            "video_title": video_title,
            "status_comments": status_comments,
            "status_transcripts": "none",
            "comments_path": "",
            "transcripts_dir": str(paths.transcripts_dir),
            "info_path": "",
            "error_message": error_message,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_sec": round((finished_at - started_at).total_seconds(), 2),
            "num_transcript_files": 0,
        }
        index_row = {
            "video_id": "",
            "url": url,
            "gpu_model": gpu_model,
            "brand": brand,
            "video_title": video_title,
            "comments_json_exists": False,
            "info_json_exists": False,
            "transcript_file_count": 0,
        }
        return ProcessResult(manifest_row, index_row, status_comments)

    logger.info("START %s (%s)", video_id, video_title)
    comments_path = paths.comments_dir / f"{video_id}.comments.json"
    info_path = paths.info_dir / f"{video_id}.info.json"
    existing_transcripts = gather_existing_transcripts(video_id)
    skip = (
        not args.force
        and comments_path.exists()
        and info_path.exists()
    )
    stderr_text = ""
    return_code = 0 if skip else 1
    if skip:
        logger.info("Skipping %s (comments + info already exist)", video_id)
    else:
        tmp_run_dir = paths.tmp_dir / video_id
        if tmp_run_dir.exists():
            shutil.rmtree(tmp_run_dir, ignore_errors=True)
        success, _, stderr_text, return_code = invoke_yt_dlp_comments(
            video_id,
            url,
            tmp_run_dir,
            args.yt_extractor_args,
            logger,
        )
        if success:
            c_path, i_path = move_comment_artifacts(
                video_id, tmp_run_dir, paths.comments_dir, paths.info_dir, logger
            )
            comments_path = c_path or comments_path
            info_path = i_path or info_path
        else:
            error_message = stderr_text.strip()[:1000]
        if tmp_run_dir.exists():
            shutil.rmtree(tmp_run_dir, ignore_errors=True)
    status_comments, comment_count = determine_comment_status(return_code, comments_path, stderr_text)
    transcripts_count = len(existing_transcripts)
    status_transcripts = "ok" if transcripts_count > 0 else "none"
    info_exists = info_path.exists()
    finished_at = datetime.now(timezone.utc)
    elapsed = round((finished_at - started_at).total_seconds(), 2)
    manifest_row = {
        "video_id": video_id,
        "url": url,
        "gpu_model": gpu_model,
        "brand": brand,
        "video_title": video_title,
        "status_comments": status_comments,
        "status_transcripts": status_transcripts,
        "comments_path": str(comments_path) if comments_path.exists() else "",
        "transcripts_dir": str(paths.transcripts_dir),
        "info_path": str(info_path) if info_exists else "",
        "error_message": error_message,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_sec": elapsed,
        "num_transcript_files": transcripts_count,
    }
    index_row = {
        "video_id": video_id,
        "url": url,
        "gpu_model": gpu_model,
        "brand": brand,
        "video_title": video_title,
        "comments_json_exists": comments_path.exists(),
        "info_json_exists": info_exists,
        "transcript_file_count": transcripts_count,
    }
    logger.info("END %s status_comments=%s duration=%.2fs", video_id, status_comments, elapsed)
    return ProcessResult(manifest_row, index_row, status_comments)


def write_manifest(manifest_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_index(index_path: Path, rows: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(index_path, index=False)


def print_summary(logger: logging.Logger, counts: Dict[str, int]) -> None:
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    logger.info("Summary: %s", summary)


def verify_smoke_test(paths: PathConfig, logger: logging.Logger) -> bool:
    checks: List[str] = []
    for required_dir in (paths.comments_dir, paths.info_dir, paths.logs_dir):
        if not required_dir.exists():
            checks.append(f"Missing directory: {required_dir}")
    if not paths.manifest_path.exists() or paths.manifest_path.stat().st_size == 0:
        checks.append("Manifest missing or empty.")
    if not paths.index_path.exists() or paths.index_path.stat().st_size == 0:
        checks.append("Index parquet missing or empty.")
    manifest_rows: List[Dict[str, str]] = []
    if paths.manifest_path.exists():
        with paths.manifest_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            manifest_rows = list(reader)
    good_comment_rows = [
        row
        for row in manifest_rows
        if row.get("status_comments") in {"ok", "none", "disabled"}
        and Path(row.get("info_path", "")).exists()
    ]
    if not good_comment_rows:
        checks.append("No manifest rows with good comment status + info JSON.")
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
    logger = configure_logging(paths.logs_dir / "comments.log")
    try:
        catalog = read_catalog(args.csv)
    except Exception as exc:
        logger.error(str(exc))
        print(str(exc), file=sys.stderr)
        return 1
    if args.max_videos:
        catalog = catalog.head(args.max_videos)
    results: List[ProcessResult] = []
    for idx, row in enumerate(catalog.to_dict(orient="records"), start=1):
        result = process_row(row, args, paths, logger)
        results.append(result)
        if idx < len(catalog):
            time.sleep(random.uniform(1, 3))
    write_manifest(paths.manifest_path, [r.manifest_row for r in results])
    write_index(paths.index_path, [r.index_row for r in results])
    print_summary(logger, summarize_status_counts(results))
    if args.smoke_test:
        ok = verify_smoke_test(paths, logger)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

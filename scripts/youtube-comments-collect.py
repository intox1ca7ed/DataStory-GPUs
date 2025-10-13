"""
Collect raw YouTube comments for every video currently listed in
`data-raw/youtube_videos.csv`.

The script relies on `yt-dlp` so no YouTube Data API key is required. Comments
are written to individual JSONL files under `data-raw/youtube-comments/`, one
file per video. Each row contains the GPU model together with comment metadata,
making downstream processing deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from yt_dlp import YoutubeDL


BASE_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = BASE_DIR / "data-raw" / "youtube_videos.csv"
OUTPUT_DIR = BASE_DIR / "data-raw" / "youtube-comments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _ts_to_iso(ts: float | int | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return ""


def _load_video_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Video catalog not found at {path}")
    df = pd.read_csv(path)
    required_cols = {"model", "video_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Video catalog missing required columns: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["video_id"]).copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df = df[df["video_id"] != ""]
    return df.drop_duplicates(subset=["video_id"])


def _yt_dlp_opts(max_comments: int | None) -> dict:
    extractor_args: dict[str, dict[str, list[str]]] = {
        "youtube": {
            "comment_sort": ["time"],
            "all_comments": ["True"],
        }
    }
    if max_comments is not None and max_comments > 0:
        extractor_args["youtube"]["max_comments"] = [str(max_comments)]

    return {
        "quiet": True,
        "skip_download": True,
        "ignoreerrors": False,
        "nocheckcertificate": True,
        "getcomments": True,
        "extractor_args": extractor_args,
    }


def fetch_comments(video_id: str, opts: dict) -> dict | None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    with YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to fetch comments for {video_id}: {exc}", file=sys.stderr)
            return None
    return info


def _existing_comment_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = data.get("comment_id")
            if cid:
                ids.add(cid)
    return ids


def _iter_comments(info: dict) -> Iterable[dict]:
    comments = info.get("comments") or []
    for comment in comments:
        if isinstance(comment, dict):
            yield comment


def _normalise_comment(comment: dict, model: str, video_id: str, retrieved_at: str) -> dict:
    parent = comment.get("parent") or ""
    is_reply = parent not in ("", "root")
    return {
        "model": model,
        "video_id": video_id,
        "comment_id": comment.get("id") or "",
        "parent_id": parent,
        "text": comment.get("text") or "",
        "author": comment.get("author") or "",
        "author_channel_id": comment.get("author_id") or "",
        "author_channel_url": comment.get("author_url") or "",
        "author_is_uploader": bool(comment.get("author_is_uploader")),
        "author_is_verified": bool(comment.get("author_is_verified")),
        "like_count": int(comment.get("like_count") or 0),
        "is_reply": is_reply,
        "published_at": _ts_to_iso(comment.get("timestamp")),
        "published_text": comment.get("time_text") or "",
        "retrieved_at": retrieved_at,
    }


def collect_all(
    max_comments: int | None = None,
    pause: float = 0.5,
    video_ids: set[str] | None = None,
    start_index: int = 0,
    force: bool = False,
) -> None:
    table = _load_video_table(CATALOG_PATH)
    if video_ids:
        table = table[table["video_id"].isin(video_ids)].reset_index(drop=True)
        missing = video_ids - set(table["video_id"])
        for vid in sorted(missing):
            print(f"[warn] video id {vid} not present in catalog; skipping", file=sys.stderr)

    opts = _yt_dlp_opts(max_comments=max_comments)
    retrieved_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    for idx, row in table.iloc[start_index:].iterrows():
        video_id = row["video_id"]
        model = row["model"]
        target_path = OUTPUT_DIR / f"{video_id}.jsonl"

        if target_path.exists() and not force:
            print(f"[skip] {video_id}: already collected ({target_path.name})", file=sys.stderr)
            continue

        existing_ids = _existing_comment_ids(target_path) if target_path.exists() else set()

        info = fetch_comments(video_id, opts)
        if not info:
            continue

        new_rows = []
        for comment in _iter_comments(info):
            cid = comment.get("id")
            if not cid or cid in existing_ids:
                continue
            new_rows.append(_normalise_comment(comment, model, video_id, retrieved_at))

        if not new_rows:
            print(f"[skip] {video_id}: no new comments", file=sys.stderr)
        else:
            with target_path.open("a", encoding="utf-8") as handle:
                for record in new_rows:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[ok] {video_id}: wrote {len(new_rows)} comments", file=sys.stderr)

        if pause > 0:
            time.sleep(pause)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw YouTube comments for catalogued GPU videos.")
    parser.add_argument(
        "--max-comments",
        type=int,
        default=None,
        help="Limit the number of comments per video (default: all available).",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Disable the short pause between videos.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Fetch comments only for the provided video IDs (can be passed multiple times).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Row offset into the CSV to resume from.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch comments even if an output file already exists.",
    )
    args = parser.parse_args()

    pause = 0.0 if args.no_sleep else 0.5
    id_filter = set(args.video_ids) if args.video_ids else None
    collect_all(
        max_comments=args.max_comments,
        pause=pause,
        video_ids=id_filter,
        start_index=max(args.start_index, 0),
        force=args.force,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import webvtt


DEFAULT_RAW_DIR = Path("data-raw")
DEFAULT_CLEAN_DIR = Path("data-clean")
COMMENT_SAMPLE_ROWS = 100_000


@dataclass
class Paths:
    raw: Path
    clean: Path
    comments_dir: Path
    info_dir: Path
    transcripts_dir: Path
    manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw YouTube comment/transcript collection into analysis-ready tables."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Root directory with raw data.")
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR, help="Output directory for clean tables.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["comments", "transcripts", "videos", "corpus", "all"],
        default=["all"],
        help="Which tables to build (default: all).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Limit processing to the first N videos for quick tests.",
    )
    parser.add_argument(
        "--write-samples",
        action="store_true",
        help=f"Emit CSV samples (first {COMMENT_SAMPLE_ROWS} rows) alongside Parquet outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def ensure_paths(raw_dir: Path, clean_dir: Path) -> Paths:
    comments_dir = raw_dir / "comments"
    info_dir = raw_dir / "info"
    transcripts_dir = raw_dir / "transcripts"
    manifest_path = raw_dir / "manifest.csv"
    clean_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        raw=raw_dir,
        clean=clean_dir,
        comments_dir=comments_dir,
        info_dir=info_dir,
        transcripts_dir=transcripts_dir,
        manifest_path=manifest_path,
    )


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        logging.warning("Manifest not found at %s; proceeding without status join.", manifest_path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(manifest_path)
    except Exception as exc:
        logging.error("Failed to read manifest: %s", exc)
        return pd.DataFrame()
    df = df.fillna("")
    return df


def extract_video_ids(manifest: pd.DataFrame, info_dir: Path, limit: Optional[int]) -> List[str]:
    ids: List[str] = []
    if not manifest.empty and "video_id" in manifest.columns:
        ids = [vid for vid in manifest["video_id"].astype(str).tolist() if vid]
    if not ids:
        logging.info("Manifest empty or missing video_id; discovering from info directory.")
        ids = [p.stem for p in info_dir.glob("*.info.json")]
    ids = sorted(dict.fromkeys(ids))  # de-duplicate while keeping order
    if limit is not None:
        ids = ids[:limit]
    logging.info("Discovered %s video IDs to process.", len(ids))
    return ids


def read_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logging.debug("JSON not found: %s", path)
        return None
    except Exception as exc:
        logging.warning("Failed to parse JSON %s: %s", path, exc)
        return None


def normalize_datetime(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    if isinstance(value, (int, float)):
        value = str(int(value))
    if not isinstance(value, str):
        value = str(value)
    if value.isdigit() and len(value) == 8:
        try:
            dt = datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
            return pd.Timestamp(dt)
        except ValueError:
            pass
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return ts if pd.notnull(ts) else None


def flatten_comment_object(obj: Dict, video_id: str, parent_id: Optional[str] = None) -> Iterable[Dict]:
    text = obj.get("text") or obj.get("content") or obj.get("body")
    if isinstance(text, dict):
        text = text.get("simpleText") or text.get("text")
    if isinstance(text, list):
        text = " ".join(str(part) for part in text if part)
    author = obj.get("author") or obj.get("author_name") or obj.get("authorText") or obj.get("uploader")
    comment_id = obj.get("id") or obj.get("commentId") or obj.get("cid")
    likes = obj.get("like_count") or obj.get("likes") or obj.get("likeCount")
    published = (
        obj.get("timestamp")
        or obj.get("time")
        or obj.get("published")
        or obj.get("published_time")
        or obj.get("publishedAt")
        or obj.get("published_time_text")
    )
    replies = None
    for key in ("replies", "comments", "children", "items"):
        candidate = obj.get(key)
        if isinstance(candidate, list):
            replies = candidate
            break
    row = {
        "video_id": video_id,
        "comment_id": comment_id,
        "parent_id": parent_id,
        "author": author,
        "text": (text or "").strip(),
        "like_count": likes,
        "published_at": normalize_datetime(published),
        "is_reply": parent_id is not None,
    }
    yield row
    if replies:
        for reply in replies:
            if isinstance(reply, dict):
                yield from flatten_comment_object(reply, video_id, parent_id=comment_id)


def extract_comments(data: object, video_id: str) -> List[Dict]:
    rows: List[Dict] = []
    queue: List[Dict] = []
    if isinstance(data, list):
        queue.extend([item for item in data if isinstance(item, dict)])
    elif isinstance(data, dict):
        for key in ("comments", "entries", "items", "contents", "response"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                queue.extend([item for item in candidate if isinstance(item, dict)])
            elif isinstance(candidate, dict):
                queue.append(candidate)
        if not queue:
            queue.append(data)
    for item in queue:
        rows.extend(flatten_comment_object(item, video_id))
    return rows


def load_comments_for_video(paths: Paths, video_id: str) -> List[Dict]:
    comment_path = paths.comments_dir / f"{video_id}.comments.json"
    if comment_path.exists():
        data = read_json(comment_path)
        if data is None:
            return []
        rows = extract_comments(data, video_id)
        for row in rows:
            row["raw_path"] = str(comment_path)
        return rows
    info_path = paths.info_dir / f"{video_id}.info.json"
    info = read_json(info_path)
    if info and "comments" in info:
        rows = extract_comments(info["comments"], video_id)
        for row in rows:
            row["raw_path"] = str(info_path)
        return rows
    logging.debug("No comments found for %s", video_id)
    return []


def load_transcripts_for_video(paths: Paths, video_id: str) -> List[Dict]:
    rows: List[Dict] = []
    pattern = paths.transcripts_dir.glob(f"{video_id}.*.vtt")
    for transcript_path in pattern:
        lang = transcript_path.stem.split(".", 1)[1] if "." in transcript_path.stem else "und"
        try:
            for caption in webvtt.read(transcript_path.as_posix()):
                text = (caption.text or "").strip()
                if not text:
                    continue
                rows.append(
                    {
                        "video_id": video_id,
                        "lang": lang,
                        "start_seconds": float(caption.start_in_seconds),
                        "end_seconds": float(caption.end_in_seconds),
                        "text": text,
                        "raw_path": str(transcript_path),
                    }
                )
        except FileNotFoundError:
            continue
        except Exception as exc:
            logging.warning("Failed to parse VTT %s: %s", transcript_path, exc)
    return rows


def parse_video_metadata(info_path: Path, video_id: str) -> Dict[str, object]:
    info = read_json(info_path)
    if not info:
        return {
            "video_id": video_id,
        }
    upload_date = normalize_datetime(info.get("upload_date") or info.get("release_date"))
    available_captions = info.get("available_captions")
    categories = info.get("categories")
    tags = info.get("tags")
    metadata = {
        "video_id": video_id,
        "title": info.get("title") or info.get("fulltitle"),
        "uploader": info.get("uploader") or info.get("uploader_id"),
        "channel": info.get("channel"),
        "channel_id": info.get("channel_id"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "categories": ",".join(categories) if isinstance(categories, list) else categories,
        "tags": ",".join(tags) if isinstance(tags, list) else tags,
        "available_captions": ",".join(available_captions) if isinstance(available_captions, list) else available_captions,
        "upload_date": upload_date,
        "webpage_url": info.get("webpage_url"),
        "info_path": str(info_path),
    }
    for date_key in ("release_timestamp", "timestamp"):
        ts = info.get(date_key)
        if ts and isinstance(ts, (int, float)):
            metadata[date_key] = pd.to_datetime(ts, unit="s", utc=True)
    return metadata


def safe_concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_table(df: pd.DataFrame, clean_dir: Path, name: str, write_samples: bool) -> Tuple[int, Optional[Path], Optional[Path]]:
    if df.empty:
        logging.info("Table %s is empty; skipping write.", name)
        return 0, None, None
    parquet_path = clean_dir / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)
    csv_path = None
    if write_samples:
        csv_path = clean_dir / f"{name}_sample.csv"
        sample_df = df.head(COMMENT_SAMPLE_ROWS)
        sample_df.to_csv(csv_path, index=False)
    logging.info("Wrote %s rows to %s", len(df), parquet_path)
    if csv_path:
        logging.info("Wrote sample to %s", csv_path)
    return len(df), parquet_path, csv_path


def build_comments(paths: Paths, video_ids: List[str]) -> pd.DataFrame:
    all_rows: List[Dict] = []
    for idx, video_id in enumerate(video_ids, 1):
        rows = load_comments_for_video(paths, video_id)
        if rows:
            all_rows.extend(rows)
        if idx % 50 == 0:
            logging.info("Processed comments for %s/%s videos.", idx, len(video_ids))
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["text"] = df["text"].astype(str).str.strip()
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    numeric_cols = ["like_count"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["text"].str.len() > 0]
    return df


def build_transcripts(paths: Paths, video_ids: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for idx, video_id in enumerate(video_ids, 1):
        rows = load_transcripts_for_video(paths, video_id)
        if rows:
            frames.append(pd.DataFrame(rows))
        if idx % 50 == 0:
            logging.info("Processed transcripts for %s/%s videos.", idx, len(video_ids))
    df = safe_concat(frames)
    if df.empty:
        return df
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    return df


def build_videos(paths: Paths, video_ids: List[str], manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    manifest_by_id = {row["video_id"]: row for _, row in manifest.iterrows()} if not manifest.empty else {}
    for video_id in video_ids:
        info_path = paths.info_dir / f"{video_id}.info.json"
        row = parse_video_metadata(info_path, video_id)
        manifest_row = manifest_by_id.get(video_id, {})
        row.update({f"manifest_{key}": value for key, value in manifest_row.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "upload_date" in df.columns:
        df["upload_date"] = pd.to_datetime(df["upload_date"], utc=True, errors="coerce")
    datetime_cols = [col for col in df.columns if col.startswith("manifest_") and col.endswith("ed_at")]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def build_corpus(comments: pd.DataFrame, transcripts: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if not comments.empty:
        comment_subset = comments.copy()
        comment_subset["source_type"] = "comment"
        comment_subset.rename(columns={"published_at": "timestamp"}, inplace=True)
        frames.append(comment_subset[["video_id", "text", "timestamp", "source_type"] + [col for col in comment_subset.columns if col not in {"video_id", "text", "timestamp", "source_type"}]])
    if not transcripts.empty:
        transcript_subset = transcripts.copy()
        transcript_subset["source_type"] = "transcript"
        transcript_subset["timestamp"] = pd.to_datetime(transcript_subset["start_seconds"], unit="s", utc=True, errors="coerce")
        frames.append(transcript_subset)
    df = safe_concat(frames)
    if df.empty:
        return df
    return df


def summarize_outputs(
    video_count: int,
    counts: Dict[str, int],
    comments: pd.DataFrame,
    transcripts: pd.DataFrame,
) -> None:
    print("=" * 60)
    print("Normalization Summary")
    print(f"Videos processed: {video_count}")
    for name, count in counts.items():
        print(f"{name.capitalize():<12}: {count:>10}")
    if not transcripts.empty and "lang" in transcripts.columns:
        lang_counts = transcripts["lang"].value_counts().head(10)
        print("\nTranscript language histogram (top 10):")
        for lang, freq in lang_counts.items():
            print(f"  {lang:<8} {freq}")
    if not comments.empty:
        non_empty = (comments["text"].astype(str).str.strip().str.len() > 0).sum()
        share = non_empty / len(comments)
        print(f"\nComments with non-empty text: {non_empty}/{len(comments)} ({share:.1%})")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    parts = set(args.parts)
    if "all" in parts:
        parts = {"comments", "transcripts", "videos", "corpus"}
    paths = ensure_paths(args.raw_dir, args.clean_dir)
    manifest = load_manifest(paths.manifest_path)
    video_ids = extract_video_ids(manifest, paths.info_dir, args.max_videos)
    if not video_ids:
        logging.error("No videos discovered; exiting.")
        return 1

    comments_df = pd.DataFrame()
    transcripts_df = pd.DataFrame()
    videos_df = pd.DataFrame()
    corpus_df = pd.DataFrame()
    counts: Dict[str, int] = {}

    if "comments" in parts:
        logging.info("Building comments table...")
        comments_df = build_comments(paths, video_ids)
        counts["comments"] = write_table(comments_df, paths.clean, "comments", args.write_samples)[0]

    if "transcripts" in parts:
        logging.info("Building transcripts table...")
        transcripts_df = build_transcripts(paths, video_ids)
        counts["transcripts"] = write_table(transcripts_df, paths.clean, "transcripts", args.write_samples)[0]

    if "videos" in parts:
        logging.info("Building videos table...")
        videos_df = build_videos(paths, video_ids, manifest)
        counts["videos"] = write_table(videos_df, paths.clean, "videos", args.write_samples)[0]

    if "corpus" in parts:
        logging.info("Building corpus table...")
        corpus_df = build_corpus(comments_df, transcripts_df)
        counts["corpus"] = write_table(corpus_df, paths.clean, "corpus", args.write_samples)[0]

    summarize_outputs(len(video_ids), counts, comments_df, transcripts_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

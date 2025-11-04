#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd

DEFAULT_CLEAN_DIR = Path("data-clean")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute quick metrics over normalized YouTube datasets."
    )
    p.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR, help="Directory with Parquet outputs.")
    p.add_argument("--topk-lang", type=int, default=10, help="Show top-K transcript languages.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"]) 
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        logging.warning("Missing dataset: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logging.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def print_comments_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("Comments: 0 rows")
        return
    total = len(df)
    has_id = df.get("comment_id").notna() if "comment_id" in df.columns else pd.Series([False]*total)
    unique_by_id = df[has_id].drop_duplicates(subset=[c for c in ["video_id","comment_id"] if c in df.columns]).shape[0]
    # Fallback uniqueness using text + timestamp within video
    subset_cols = [c for c in ["video_id","text","published_at"] if c in df.columns]
    if subset_cols:
        unique_fallback = df[~has_id].drop_duplicates(subset=subset_cols).shape[0]
    else:
        unique_fallback = 0
    top_level = int((~df.get("is_reply", pd.Series([False]*total))).sum()) if "is_reply" in df.columns else None
    replies = int(df.get("is_reply", pd.Series([False]*total)).sum()) if "is_reply" in df.columns else None

    non_empty = (df["text"].astype(str).str.strip().str.len() > 0).sum() if "text" in df.columns else total
    print("Comments")
    print(f"  rows            : {total}")
    print(f"  unique_by_id    : {unique_by_id}")
    print(f"  unique_fallback : {unique_fallback}")
    print(f"  unique_est_total: {unique_by_id + unique_fallback}")
    if top_level is not None:
        print(f"  top_level       : {top_level}")
        print(f"  replies         : {replies}")
    print(f"  non_empty_text  : {non_empty}")


def print_transcript_stats(df: pd.DataFrame, topk: int) -> None:
    if df.empty:
        print("Transcripts: 0 rows")
        return
    total = len(df)
    dur = 0.0
    if {"start_seconds","end_seconds"}.issubset(df.columns):
        dur = (df["end_seconds"] - df["start_seconds"]).clip(lower=0).sum()
    print("Transcripts")
    print(f"  segments : {total}")
    print(f"  hours    : {dur/3600:.2f}")
    if "lang" in df.columns:
        vc = df["lang"].value_counts().head(topk)
        print("  lang topK:")
        for lang, cnt in vc.items():
            print(f"    {lang:<8} {cnt}")


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    clean = args.clean_dir
    comments = read_parquet(clean / "comments.parquet")
    transcripts = read_parquet(clean / "transcripts.parquet")
    videos = read_parquet(clean / "videos.parquet")
    corpus = read_parquet(clean / "corpus.parquet")

    print("="*60)
    print("YT Dataset Metrics (data-clean)")
    if not videos.empty:
        print(f"Videos     : {len(videos)}")
    if not corpus.empty:
        print(f"Corpus rows: {len(corpus)}")
    print_comments_stats(comments)
    print_transcript_stats(transcripts, args.topk_lang)
    print("="*60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

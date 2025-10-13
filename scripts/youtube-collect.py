import os
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build
from langdetect import detect

API_KEY = os.environ["YOUTUBE_API_KEY"]
yt = build("youtube", "v3", developerKey=API_KEY)

CACHE_DIR = Path("data-raw/cache/youtube_search")
VIDEO_CACHE_DIR = Path("data-raw/cache/youtube_videos")
PROGRESS_PATH = Path("data-raw/youtube_progress.json")
MODELS_PER_RUN = int(os.getenv("YOUTUBE_MAX_MODELS_PER_RUN", "10"))

for directory in (CACHE_DIR, VIDEO_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

BASE_KWS = ["review", "benchmark", "performance", "analysis", "testing", "vs", "gameplay"]


def _search_cache_path(query, published_after, published_before, max_results, video_duration):
    raw = "|".join([
        query,
        published_after or "",
        published_before or "",
        str(max_results),
        video_duration or "",
    ])
    return CACHE_DIR / f"{hashlib.md5(raw.encode('utf-8')).hexdigest()}.json"


def search_videos(query, published_after=None, published_before=None, max_results=50, video_duration=None):
    cache_path = _search_cache_path(query, published_after, published_before, max_results, video_duration)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached.get("items", [])
        except json.JSONDecodeError:
            cache_path.unlink(missing_ok=True)

    params = {
        "part": "id,snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": "viewCount",
        "relevanceLanguage": "en",
        "publishedAfter": published_after,
        "publishedBefore": published_before,
        "regionCode": "US",
        "safeSearch": "none",
    }
    if video_duration:
        params["videoDuration"] = video_duration

    res = yt.search().list(**params).execute()
    cache_path.write_text(json.dumps(res), encoding="utf-8")
    return res.get("items", [])


def _video_cache_path(video_id):
    sanitized = video_id.replace("/", "_")
    return VIDEO_CACHE_DIR / f"{sanitized}.json"


def enrich_stats(video_ids):
    if not video_ids:
        return {}

    out = {}
    pending = []
    for vid in video_ids:
        path = _video_cache_path(vid)
        if path.exists():
            try:
                out[vid] = json.loads(path.read_text(encoding="utf-8"))
                continue
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
        pending.append(vid)

    for start in range(0, len(pending), 50):
        chunk = pending[start:start + 50]
        stats = yt.videos().list(
            part="statistics,contentDetails,snippet",
            id=",".join(chunk)
        ).execute()
        for it in stats.get("items", []):
            payload = {
                "viewCount": int(it["statistics"].get("viewCount", 0)),
                "duration": it["contentDetails"]["duration"],
                "title": it["snippet"]["title"],
                "description": it["snippet"].get("description", ""),
                "channelId": it["snippet"]["channelId"],
                "channelTitle": it["snippet"]["channelTitle"],
                "publishedAt": it["snippet"]["publishedAt"],
            }
            out[it["id"]] = payload
            _video_cache_path(it["id"]).write_text(json.dumps(payload), encoding="utf-8")
    return out


def _normalize_synonyms(synonyms_str):
    if not isinstance(synonyms_str, str):
        return []
    return [s.strip() for s in synonyms_str.split(";") if s.strip()]


def build_queries(model, query_terms, synonyms_str):
    keywords = BASE_KWS + ([w for w in str(query_terms).split() if w] if query_terms else [])
    keywords = list(dict.fromkeys(keywords))
    suffix = " ".join(keywords).strip()

    queries = [f"\"{model}\" {suffix}".strip()]

    synonyms = list(dict.fromkeys(_normalize_synonyms(synonyms_str)))
    if synonyms:
        syn_clause = " OR ".join(f"\"{s}\"" for s in synonyms[:3])
        queries.append(f"({syn_clause}) {suffix}".strip())

    return [q for q in queries[:2] if q]


def _time_windows(launch_date):
    launch = pd.to_datetime(launch_date, errors="coerce")
    if pd.isna(launch):
        return [(None, None)]
    return [
        (launch - pd.Timedelta(days=30), launch + pd.Timedelta(days=150)),
        (launch - pd.Timedelta(days=365), launch + pd.Timedelta(days=365)),
        (None, None),
    ]


def _format_window(ts):
    if ts is None:
        return None
    return f"{ts:%Y-%m-%dT00:00:00Z}"


def _select_videos(model, meta, k):
    rows = []
    channel_counts = {}
    seen_vids = set()
    for vid, m in sorted(meta.items(), key=lambda kv: -kv[1]["viewCount"]):
        if vid in seen_vids:
            continue
        text = (m.get("title", "") + " " + m.get("description", ""))[:5000]
        try:
            if text.strip() and detect(text) != "en":
                continue
        except Exception:
            # If detection fails, fall back to accepting the video.
            pass
        cc = channel_counts.get(m.get("channelId"), 0)
        if cc >= (2 if len(rows) < k else 1):
            continue

        rows.append({
            "model": model,
            "video_id": vid,
            "title": m.get("title", ""),
            "channel_id": m.get("channelId", ""),
            "channel_title": m.get("channelTitle", ""),
            "published_at": m.get("publishedAt", ""),
            "views": m.get("viewCount", 0),
        })
        channel_counts[m.get("channelId")] = cc + 1
        seen_vids.add(vid)
        if len(rows) >= k:
            break
    return rows


def collect_for_model(model, launch_date, query_terms, synonyms="", k=10):
    queries = build_queries(model, query_terms, synonyms or "")
    candidate_ids = []
    rows = []

    windows = _time_windows(launch_date)
    for window_idx, (start_ts, end_ts) in enumerate(windows):
        published_after = _format_window(start_ts)
        published_before = _format_window(end_ts)

        for q in queries:
            items = search_videos(
                q,
                published_after=published_after,
                published_before=published_before,
                max_results=50,
            )
            for it in items:
                vid = (it.get("id") or {}).get("videoId")
                if vid:
                    candidate_ids.append(vid)
            if len(candidate_ids) >= k * 8:
                break
        unique_ids = list(dict.fromkeys(candidate_ids))
        meta = enrich_stats(unique_ids)
        rows = _select_videos(model, meta, k)

        if len(rows) >= k or window_idx == len(windows) - 1 or len(candidate_ids) >= k * 8:
            break
        # Brief pause before widening window to respect API rate limits slightly.
        time.sleep(0.1)

    return rows


def _load_progress():
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            PROGRESS_PATH.unlink(missing_ok=True)
    return {"completed_models": []}


def _save_progress(progress):
    progress["last_run"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main():
    cat = pd.read_csv("gpu_catalog.csv")
    progress = _load_progress()
    completed = set(progress.get("completed_models", []))

    remaining = [r for _, r in cat.iterrows() if r["model"] not in completed]
    if not remaining:
        print("All models already collected.")
        return

    max_models = max(MODELS_PER_RUN, 1)
    output_path = Path("data-raw/youtube_videos.csv")
    header_needed = not output_path.exists()

    to_append = []
    processed_now = 0

    for r in remaining:
        if processed_now >= max_models:
            break
        rows = collect_for_model(
            r["model"], r["launch_date"], r.get("query_terms", ""),
            synonyms=r.get("synonyms", ""), k=10
        )
        if rows:
            to_append.extend(rows)
        completed.add(r["model"])
        progress["completed_models"] = sorted(completed)
        _save_progress(progress)
        processed_now += 1
        print(f"{r['model']}: collected {len(rows)} rows")
        time.sleep(0.2)

    if to_append:
        pd.DataFrame(to_append).to_csv(
            output_path, mode="a", header=header_needed, index=False
        )
        print(f"Appended {len(to_append)} rows to {output_path}")
    else:
        print("No new rows collected this run.")

    remaining_models = len(cat) - len(completed)
    if remaining_models > 0:
        print(f"{remaining_models} models remain. Re-run after quota reset to continue.")


if __name__ == "__main__":
    main()

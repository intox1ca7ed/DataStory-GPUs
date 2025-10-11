import time
import pandas as pd
from pathlib import Path
import importlib.util

_THIS_DIR = Path(__file__).resolve().parent
_CATALOG_PATH = Path("gpu_catalog.csv")
_OUTPUT_PATH = Path("data-raw/youtube_videos_nvidia_30xx_test.csv")

def _load_youtube_module():
    spec = importlib.util.spec_from_file_location(
        "youtube_collect", _THIS_DIR / "youtube-collect.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def _filter_nvidia_30_series(df: pd.DataFrame) -> pd.DataFrame:
    vendor_col = next((c for c in ("vendor", "brand", "manufacturer") if c in df.columns), None)
    if vendor_col:
        df = df[df[vendor_col].astype(str).str.contains("nvidia", case=False, na=False)]
    else:
        df = df[df["model"].astype(str).str.contains("nvidia", case=False, na=False)]
    mask_30_series = df["model"].astype(str).str.contains(r"30\d{2}", case=False, na=False)
    return df.loc[mask_30_series].copy()

def _ensure_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found at {path}")
    df = pd.read_csv(path)
    if "model" not in df.columns or "launch_date" not in df.columns:
        raise ValueError("Catalog must include 'model' and 'launch_date' columns")
    return df

def main():
    module = _load_youtube_module()
    catalog = _ensure_catalog(_CATALOG_PATH)
    subset = _filter_nvidia_30_series(catalog)
    if subset.empty:
        print("No NVIDIA 30-series entries detected in the catalog; nothing to test.")
        return

    collected = []
    start_total = time.perf_counter()
    for _, row in subset.iterrows():
        launched = row["launch_date"]
        model = row["model"]
        query_terms = row.get("query_terms", "")
        synonyms = row.get("synonyms", "")
        start_model = time.perf_counter()
        rows = module.collect_for_model(model, launched, query_terms, synonyms=synonyms)
        duration_model = time.perf_counter() - start_model
        collected.extend(rows)
        print(f"{model}: {len(rows)} rows in {duration_model:.2f}s")
        time.sleep(0.2)

    total_duration = time.perf_counter() - start_total
    if collected:
        pd.DataFrame(collected).to_csv(_OUTPUT_PATH, index=False)
        print(f"Wrote {len(collected)} rows to {_OUTPUT_PATH}")
    else:
        print("Collection returned no rows.")
    print(f"Total runtime {total_duration:.2f}s for {len(subset)} models")

if __name__ == "__main__":
    main()

import os, pandas as pd, requests, urllib.parse, time
SERPAPI_KEY = os.environ["SERPAPI_KEY"]
DOMAINS = ["anandtech.com","techpowerup.com","tomshardware.com","pcgamer.com",
           "eurogamer.net","guru3d.com","techspot.com","notebookcheck.net","digitalfoundry.net"]

def search_articles(model, k=3):
    rows = []
    for d in DOMAINS:
        q = f'site:{d} "{model}" review'
        url = f"https://serpapi.com/search.json?engine=google&q={urllib.parse.quote(q)}&num=5&api_key={SERPAPI_KEY}"
        r = requests.get(url, timeout=20).json()
        for it in r.get("organic_results", []):
            rows.append({"model": model, "title": it.get("title"), "link": it.get("link"), "source": d})
        time.sleep(0.5)
        if len(rows) >= k: break
    return rows[:k]

def main():
    cat = pd.read_csv("gpu_catalog.csv")
    out = []
    for _, r in cat.iterrows():
        out.extend(search_articles(r["model"], k=3))
    pd.DataFrame(out).to_csv("data-raw/review_articles.csv", index=False)

if __name__ == "__main__":
    main()

import pandas as pd

#we use this script to generate yt-dlp commands
#it will output a .bat file with all yt-dlp commands to download comments
#the .bat file itself will take 30 to 40 minutes to run

df = pd.read_csv("../gpu-list-links.csv") 

with open("yt-download.bat", "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        brand = row['Brand'].replace(' ', '_')
        model = row['GPU Model'].replace(' ', '_')
        video_id = row['YouTube URL'].split('=')[-1]
        url = row["YouTube URL"]
        out_fn = f"../data-raw/{brand}_{model}_{video_id}.%(ext)s"
        cmd = f'yt-dlp --write-comments --skip-download "{url}" -o "{out_fn}"\n'
        f.write(cmd)
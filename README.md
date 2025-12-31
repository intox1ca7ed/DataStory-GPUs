# GPU Launch Reactions on YouTube — How to Read This Project

This project is designed to be consumed as a **standalone data story**. The **presentation is the main reading source**. The notebook is included as supporting evidence and reproducibility.

## 1) Start with the presentation (main source)
Open the slide deck first. It is written to be **holistic and independent**, meaning you can understand the story without a live presenter.

## 2) Use the notebook to verify and explore (supporting evidence)
The Jupyter notebook contains the full analysis pipeline and all intermediate steps used to generate the figures and numbers shown in the presentation.

Use the notebook when you want to:
- Check exact calculations behind a chart or statistic
- See additional plots that didn’t fit into the deck
- Understand cleaning/normalization steps
- Re-run the analysis

Important methodological note: YouTube comment timestamps were not available, so time-series plots use **video upload month as a proxy** for when comments occurred. This is acceptable for the launch-window design but should be interpreted accordingly.

## 3) Repository structure (what each file is for)
- `presentation.pdf` the main narrative artifact
- `01_gpu_eda_main-report.ipynb` (and/or cleaned version): analysis + figures + supporting outputs
- `01_gpu_eda_main-report.pdf`: static snapshot of the notebook for quick viewing
- `youtube_videos.csv`: list of videos included and their metadata
- `yt_fetch_comments.py`, `yt_fetch_transcripts.py`: data collection scripts
- `yt_normalize.py`: cleaning/normalization utilities
- `yt_stats.py`: descriptive statistics / summary helpers
- `figures/`: exported visuals used in slides

## 4) What to pay attention to
The core message is not “which GPU is objectively best,” but how the community behaves around launches:
- Attention is highly skewed (a few videos/models dominate discussion)
- Price/value dominates conversation across generations
- Comments are consistently more positive than creator transcripts
- External context (e.g., 2021–2022 crypto boom) visibly affects discussion topics

---
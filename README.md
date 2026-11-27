# 🎬 CineSent — IMDB Sentiment Analysis (TF-IDF vs fine-tuned DistilBERT)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/License-MIT-green)

Binary sentiment classification on **50,000 IMDB movie reviews**, done properly: leak-free splits, tuned classical
models, a fine-tuned **DistilBERT**, honest statistics (95% bootstrap CIs, paired McNemar tests, calibration),
word-level explainability, error analysis — and a polished **Streamlit app** to play with all of it.

> Best test accuracy: **93.14 %** (ensemble of Logistic Regression + DistilBERT).
> The deployed live model, TF-IDF + Logistic Regression, reaches **91.24 %** on CPU in milliseconds with a 3.8 MB model file.

---

## ✨ App features

| Tab | What you get |
|---|---|
| 🔮 **Live Analyzer** | Type any review → verdict, confidence gauge, **word-by-word highlighting**, top-term chart, exact log-odds waterfall, mixed-signal / uncertainty flags, 10 one-click edge cases |
| 📦 **Batch Lab** | Upload a CSV or paste lines → distribution charts, downloadable predictions; add a ground-truth column to get accuracy, macro-F1 and a confusion matrix |
| 🎞️ **Review Explorer** | Browse 300 held-out test reviews, filter by mistakes / model disagreement / confident errors, compare LR vs DistilBERT vs Ensemble |
| 🏆 **Model Arena** | Interactive leaderboard with 95 % CIs, paired significance tests, speed-vs-size, calibration, DistilBERT training curves |
| 🧠 **Explainability** | Strongest words & phrases, **term lookup** (see how *good*, *not good*, *not bad* are weighted), top bigrams |
| 🔬 **Errors & Data** | Accuracy by review length, hand-written stress tests (negation, sarcasm, contrast), EDA and split overview |
| ℹ️ **About** | Pipeline, design decisions and known limitations |

The sidebar controls the **decision threshold**, an **uncertainty band**, and how many terms to show.

## 📊 Results (5,000-review test set, evaluated once)

| Model | Accuracy (95 % CI) | Macro F1 | ROC-AUC | Log-loss |
|---|---|---|---|---|
| **Ensemble (LR + DistilBERT)** | **0.9314** (0.924–0.938) | 0.9314 | 0.9793 | 0.1861 |
| DistilBERT (fine-tuned) | 0.9164 (0.909–0.924) | 0.9164 | 0.9739 | 0.2335 |
| Logistic Regression (TF-IDF 1-2 gram) | 0.9124 (0.904–0.920) | 0.9124 | 0.9724 | 0.2171 |
| Linear SVM (calibrated) | 0.9098 (0.902–0.918) | 0.9098 | 0.9707 | 0.2188 |
| Naive Bayes | 0.8932 (0.885–0.902) | 0.8932 | 0.9555 | 0.2725 |
| Dummy (prior) | 0.5018 (0.488–0.517) | 0.3341 | 0.5000 | 0.6931 |

**What the numbers say**

- DistilBERT vs Logistic Regression: **+0.4 points, p = 0.37** (exact McNemar) → not statistically distinguishable on 5,000 reviews.
- Ensemble vs DistilBERT: **+1.5 points, p < 0.0001** → averaging the two models is the only clearly significant gain; their mistakes differ.
- Adding **bigrams** to TF-IDF improved validation accuracy from 90.52 % to 92.24 % (this is how a bag-of-words model handles *not good*).
- LR is **3.3× faster** (2,508 reviews/s on CPU vs 759/s for DistilBERT on a T4 GPU) and **67× smaller** (3.8 MB vs 255 MB), and its probabilities are better calibrated — that is why it powers the live app.
- Both models are wrong together on 203 / 5,000 reviews (4.1 %): ambiguous text, plot summaries and label noise.

![Model comparison](assets/model_comparison.png)

## 🗂️ Project structure

```
sentiment-analysis-imdb/
├── app.py                        # Streamlit app (all tabs)
├── requirements.txt              # app dependencies (scikit-learn pinned to the training version)
├── requirements-transformer.txt  # optional: torch + transformers for live DistilBERT
├── models/
│   └── tfidf_lr.joblib           # TF-IDF + Logistic Regression pipeline (raw text → probability)
├── data/
│   └── demo_reviews.csv          # 300 held-out test reviews with LR / DistilBERT probabilities
├── artifacts/                    # metrics exported by the notebook (CSV / JSON)
├── assets/                       # figures exported by the notebook (PNG)
├── notebook/
│   └── sentiment-analysis.ipynb  # full pipeline: cleaning → EDA → models → DistilBERT → analysis
├── tests/test_app.py             # Streamlit smoke tests (pytest)
└── .streamlit/config.toml        # dark theme
```

## 🚀 Quick start

```bash
git clone <your-repo-url>
cd sentiment-analysis-imdb
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> `scikit-learn==1.6.1` is pinned because `models/tfidf_lr.joblib` was pickled with that version.

### Optional: live DistilBERT + ensemble

The app ships without the 255 MB DistilBERT weights (too large for a normal git repo). Everything still works:
DistilBERT scores for the 300 demo reviews and the 10 edge-case sentences are precomputed from the notebook.
To score **any** text with DistilBERT and the ensemble:

1. Run the notebook on Kaggle (GPU + Internet on). It saves the model to `/kaggle/working/distilbert_sentiment`.
2. Zip and download it: `!cd /kaggle/working && zip -r distilbert_sentiment.zip distilbert_sentiment`
3. Unzip into `models/distilbert_sentiment/` (folder with `config.json`, `model.safetensors`, tokenizer files).
4. `pip install -r requirements-transformer.txt` and restart the app.

Prefer not to keep the weights locally? Upload them to the Hugging Face Hub and point the app at the repo id:

```bash
export DISTILBERT_PATH="your-username/imdb-distilbert"      # or a local folder
streamlit run app.py
```

(On Streamlit Community Cloud put `DISTILBERT_PATH = "your-username/imdb-distilbert"` in *Secrets* and add
`torch` + `transformers` to `requirements.txt`.)

### Tests

```bash
pip install pytest
pytest -q
```

## ☁️ Deploy (Streamlit Community Cloud)

1. Push this folder to GitHub (the model file is only ~4 MB).
2. On <https://share.streamlit.io> choose the repo, branch `main`, main file `app.py`.
3. Deploy. No secrets are needed for the default (Logistic Regression) mode.

## 🧪 Methodology

1. **Data quality** — 50,000 reviews → 418 exact duplicates removed → 49,582; HTML entities and `<br />` tags stripped.
2. **Leak-free split** — stratified 39,582 train / 5,000 validation / 5,000 test with a fixed seed. Hyper-parameters are
   chosen on validation; the test set is used once.
3. **Classical models** — one TF-IDF vectoriser (1-2 grams, `min_df=3`, `max_df=0.9`, sublinear TF, 200k features) feeding
   Naive Bayes (α=0.1), Linear SVM (C=0.3, probability-calibrated) and Logistic Regression (C=10).
4. **DistilBERT** — `distilbert-base-uncased`, custom PyTorch loop, fp16, linear warm-up/decay, max length 256, 2 epochs,
   best-epoch checkpoint on validation (val acc 92.04 %).
5. **Evaluation** — accuracy with 95 % bootstrap CI, macro-F1, ROC-AUC, log-loss, calibration curves, speed and size,
   exact McNemar tests between models.
6. **Explainability** — LR is additive, so each prediction decomposes exactly into `bias + Σ weight × TF-IDF`;
   the app shows this per word and per bigram.
7. **Error analysis** — accuracy by review length, confident mistakes, where models disagree, hand-written edge cases.

## ⚠️ Limitations

- Trained on English **movie reviews**; expect weaker results on other domains, languages or emoji-heavy text.
- A bag-of-words model cannot truly read sarcasm, long-range negation or neutral text (e.g. it rates
  *"nothing special, nothing terrible"* as strongly negative). The app flags mixed and low-confidence cases.
- DistilBERT reads at most 256 tokens; 40.7 % of test reviews are longer than that.
- The task is binary — genuinely mixed reviews have to land on one side.

## 📚 Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
(Maas et al., 2011). The raw dataset is not redistributed here; the notebook finds it automatically on Kaggle.

## 📄 License

MIT — see [LICENSE](LICENSE).

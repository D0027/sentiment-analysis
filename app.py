"""
CineSent · IMDB Sentiment Intelligence
======================================
Production-style Streamlit app on top of the sentiment-analysis notebook.

* Live inference with the TF-IDF (1-2 gram) + Logistic Regression pipeline (CPU, ~ms)
* Word-level explanations: every word / bigram is coloured by how much it pushed the prediction
* Optional live DistilBERT + ensemble (drop the fine-tuned weights into models/distilbert_sentiment)
* Batch lab (CSV upload / paste) with optional ground-truth evaluation
* Review explorer over 300 held-out test reviews with precomputed LR / DistilBERT / ensemble scores
* Model arena with 95% bootstrap CIs, paired tests, speed vs size, calibration
* Global explainability, error analysis, edge cases and EDA

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import importlib.util
import io
import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tfidf_lr.joblib"
DISTIL_DIR = ROOT / "models" / "distilbert_sentiment"
ART, ASSETS, DATA = ROOT / "artifacts", ROOT / "assets", ROOT / "data"

GOLD, POS, NEG = "#F5C518", "#22C55E", "#F43F5E"
BLUE, VIOLET, MUTED, INK = "#60A5FA", "#A78BFA", "#8B95A9", "#E8ECF4"
MAX_BATCH = 5000
MAX_LEN = 256
LR_NAME, DB_NAME, ENS_NAME = "Logistic Regression", "DistilBERT", "Ensemble"

# Paired exact McNemar tests, copied from the notebook output (test set, n = 5,000)
PAIRED_TESTS = [
    ("DistilBERT vs Logistic Regression", 235, 215, "0.370", "Not significant"),
    ("Ensemble vs DistilBERT", 107, 32, "< 0.0001", "Significant"),
]

QUICK_EXAMPLES = [
    ("✨ Masterpiece", "An absolute masterpiece. I was moved to tears."),
    ("💀 Worst movie", "Worst movie I have seen in years, a complete waste of time."),
    ("🚫 Negation", "The movie was not good at all."),
    ("🔄 Plot twist", "I expected to hate it, but I ended up loving every minute."),
    ("🙄 Sarcasm", "Great, another boring sequel. Just what the world needed."),
    ("⚖️ Mixed", "The acting was brilliant but the story made no sense."),
    ("😐 Neutral", "It was fine. Nothing special, nothing terrible."),
    ("👎 Not recommended", "I would not recommend this to anyone, even for free."),
    ("👍 Not bad", "Not bad at all, actually quite enjoyable!"),
    ("🎞️ Predictable", "The plot was predictable, yet somehow I couldn't stop watching."),
]
EDGE_TAGS = ["Clear positive", "Clear negative", "Negation", "Reversal", "Sarcasm",
             "Mixed opinion", "Neutral", "Negation", "Negation (positive)", "Contrast"]
DEFAULT_TEXT = ("I expected to hate this movie, but the performances were brilliant and the story genuinely "
                "moved me. A few scenes dragged, yet I loved every minute of it.")

st.set_page_config(page_title="CineSent · IMDB Sentiment Intelligence", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
 background:radial-gradient(1100px 560px at 8% -8%,rgba(245,197,24,.11),transparent 60%),
 radial-gradient(900px 520px at 100% 0%,rgba(96,165,250,.11),transparent 58%),
 radial-gradient(800px 600px at 50% 120%,rgba(167,139,250,.08),transparent 60%),#0B0F1A;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1280px;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0E1424 0%,#0B0F1A 100%);border-right:1px solid rgba(255,255,255,.06);}
h1,h2,h3,h4{letter-spacing:-.01em;}
h3{font-weight:700;}

.hero{position:relative;overflow:hidden;padding:30px 34px;border-radius:24px;margin-bottom:16px;
 border:1px solid rgba(255,255,255,.09);
 background:linear-gradient(135deg,rgba(245,197,24,.14),rgba(96,165,250,.08) 55%,rgba(167,139,250,.12));
 box-shadow:0 20px 60px rgba(0,0,0,.35);}
.hero:after{content:"🎬";position:absolute;right:28px;top:6px;font-size:120px;opacity:.07;transform:rotate(-10deg);}
.hero .eyebrow{font-size:.74rem;letter-spacing:.22em;text-transform:uppercase;color:#F5C518;font-weight:700;}
.hero h1{margin:.25rem 0 .35rem 0;font-size:2.55rem;font-weight:800;line-height:1.1;
 background:linear-gradient(90deg,#fff 0%,#F5C518 55%,#FDE68A 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{margin:0;color:#B7C0D3;font-size:1.02rem;max-width:760px;}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:0 0 18px 0;}
.kpi{padding:16px 18px;border-radius:18px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.07);backdrop-filter:blur(6px);}
.kpi .v{font-size:1.65rem;font-weight:800;color:#fff;line-height:1.15;}
.kpi .l{font-size:.74rem;text-transform:uppercase;letter-spacing:.09em;color:#8B95A9;font-weight:600;margin-top:2px;}
.kpi .s{font-size:.78rem;color:#F5C518;margin-top:3px;font-weight:600;}

.card{padding:18px 20px;border-radius:18px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.07);margin-bottom:12px;}
.card h4{margin:0 0 6px 0;font-size:1rem;font-weight:700;}
.card p,.card li{color:#B7C0D3;font-size:.93rem;line-height:1.55;}
.muted{color:#8B95A9;font-size:.85rem;}

.verdict{padding:20px 22px;border-radius:20px;border:1px solid rgba(255,255,255,.09);margin-bottom:8px;}
.verdict .big{font-size:1.9rem;font-weight:800;line-height:1.15;}
.verdict .sub{color:#B7C0D3;font-size:.92rem;margin-top:4px;}
.chip{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.76rem;font-weight:700;margin:8px 6px 0 0;
 border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.05);color:#D5DBE8;}
.chip.g{color:#22C55E;border-color:rgba(34,197,94,.45);background:rgba(34,197,94,.10);}
.chip.r{color:#F43F5E;border-color:rgba(244,63,94,.45);background:rgba(244,63,94,.10);}
.chip.y{color:#F5C518;border-color:rgba(245,197,24,.45);background:rgba(245,197,24,.10);}
.chip.b{color:#60A5FA;border-color:rgba(96,165,250,.45);background:rgba(96,165,250,.10);}

.mcard{padding:14px 16px;border-radius:16px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.07);height:100%;}
.mcard .n{font-size:.78rem;color:#8B95A9;text-transform:uppercase;letter-spacing:.08em;font-weight:700;}
.mcard .p{font-size:1.7rem;font-weight:800;margin:2px 0;}
.mcard .d{font-size:.82rem;color:#B7C0D3;}

.hl{padding:16px 18px;border-radius:16px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
 line-height:2.15;font-size:1.02rem;max-height:340px;overflow-y:auto;color:#E8ECF4;}
.hl .t{padding:2px 4px;border-radius:6px;}
.legend{display:flex;align-items:center;gap:10px;margin-top:8px;font-size:.78rem;color:#8B95A9;}
.legend .bar{flex:0 0 190px;height:8px;border-radius:99px;background:linear-gradient(90deg,#F43F5E,rgba(255,255,255,.12) 50%,#22C55E);}

.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;margin:6px 0 4px 0;}
.step{flex:1 1 150px;padding:14px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);}
.step b{display:block;color:#F5C518;font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;}
.step span{color:#C5CCDB;font-size:.88rem;}

button[data-baseweb="tab"]{font-weight:600;padding:10px 14px;}
button[data-baseweb="tab"][aria-selected="true"]{color:#F5C518;}
div[data-baseweb="tab-highlight"]{background-color:#F5C518 !important;}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid rgba(255,255,255,.14);font-weight:600;transition:all .15s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#F5C518;color:#F5C518;transform:translateY(-1px);}
.stTextArea textarea{border-radius:14px;font-size:1rem;line-height:1.55;}
div[data-testid="stImage"] img{border-radius:14px;}
div[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def h(s: str) -> str:
    """Flatten an HTML snippet so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def md(s: str) -> None:
    st.markdown(h(s), unsafe_allow_html=True)


def show_fig(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def style_fig(fig: go.Figure, height: int = 360, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=10, t=44, b=10), hoverlabel=dict(bgcolor="#121A2B", font_size=13),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Text helpers (identical preprocessing to the notebook)
# ──────────────────────────────────────────────────────────────────────────────
TOKEN_RE = re.compile(r"(?u)\b\w[\w']*\b")


def clean_text(t: str) -> str:
    t = html.unescape(str(t))
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def pct(p: float, d: int = 1) -> str:
    return f"{100 * p:.{d}f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TF-IDF + Logistic Regression…")
def load_bundle() -> dict:
    pipe = joblib.load(MODEL_PATH)
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    return {
        "pipe": pipe, "vec": vec, "clf": clf, "names": names,
        "coef": clf.coef_.ravel().astype(np.float64), "bias": float(clf.intercept_[0]),
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "is_bigram": np.char.find(names.astype(str), " ") >= 0,
        "padded": np.char.add(np.char.add(" ", names.astype(str)), " "),
        "size_mb": MODEL_PATH.stat().st_size / 1024 / 1024,
    }


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads((ART / "model_meta.json").read_text())


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    d = pd.read_csv(DATA / "demo_reviews.csv")
    d["ens_prob"] = (d["lr_prob"] + d["distilbert_prob"]) / 2
    return d


@st.cache_data(show_spinner=False)
def known_distilbert() -> dict:
    """Notebook-time DistilBERT probabilities for the demo + edge-case reviews (no weights needed)."""
    known = {clean_text(t): float(p) for t, p in zip(load_demo()["text"], load_demo()["distilbert_prob"])}
    ce = load_csv(str(ART / "custom_examples.csv"))
    known.update({clean_text(t): float(p) for t, p in zip(ce["review"], ce["DistilBERT P(pos)"])})
    return known


@st.cache_data(show_spinner=False)
def known_truth() -> dict:
    d = load_demo()
    return {clean_text(t): int(y) for t, y in zip(d["text"], d["label"])}


# ── optional live DistilBERT ──────────────────────────────────────────────────
def distil_source() -> str | None:
    src = os.environ.get("DISTILBERT_PATH")
    if not src:
        try:
            src = st.secrets.get("DISTILBERT_PATH")
        except Exception:
            src = None
    if src:
        return str(src)
    if DISTIL_DIR.exists() and any(DISTIL_DIR.iterdir()):
        return str(DISTIL_DIR)
    return None


def distil_libs_ok() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


@st.cache_resource(show_spinner="Loading fine-tuned DistilBERT…")
def load_distilbert(src: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(dev), dev


def distil_predict(texts: list[str], src: str, bs: int = 32, progress=None) -> np.ndarray:
    import torch

    tok, model, dev = load_distilbert(src)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            out[idx] = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
            if progress:
                progress(min(1.0, (s + bs) / len(order)))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Model logic
# ──────────────────────────────────────────────────────────────────────────────
def explain(text: str, B: dict) -> dict:
    """Exact additive decomposition of the LR log-odds into per-term contributions,
    plus a per-word score used for highlighting (bigram weight is shared by its two words)."""
    cleaned = clean_text(text)
    X = B["vec"].transform([cleaned])
    coef, vocab, names = B["coef"], B["vocab"], B["names"]
    contrib = {int(i): float(v) * coef[i] for i, v in zip(X.indices, X.data)}

    toks = [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(cleaned)]
    low = [t[0].lower() for t in toks]
    feats, occ = [], {}
    for i, t in enumerate(low):
        u = vocab.get(t)
        b = vocab.get(f"{t} {low[i + 1]}") if i + 1 < len(low) else None
        feats.append((u, b))
        for j in (u, b):
            if j is not None and j in contrib:
                occ[j] = occ.get(j, 0) + 1
    scores = np.zeros(len(toks))
    for i, (u, b) in enumerate(feats):
        if u is not None and u in contrib:
            scores[i] += contrib[u] / occ[u]
        if b is not None and b in contrib:
            share = contrib[b] / occ[b] / 2
            scores[i] += share
            scores[i + 1] += share

    terms = pd.DataFrame({"term": [names[j] for j in contrib], "contribution": list(contrib.values())})
    terms = terms.reindex(terms["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    pos_sum = float(terms.loc[terms.contribution > 0, "contribution"].sum())
    neg_sum = float(terms.loc[terms.contribution < 0, "contribution"].sum())
    logit = B["bias"] + pos_sum + neg_sum
    return {"cleaned": cleaned, "tokens": toks, "scores": scores, "terms": terms,
            "pos_sum": pos_sum, "neg_sum": neg_sum, "bias": B["bias"], "logit": logit,
            "n_words": len(toks), "n_known": len(contrib)}


def highlight_html(ex: dict) -> str:
    cleaned, toks, scores = ex["cleaned"], ex["tokens"], ex["scores"]
    ref = max(1.0, float(np.abs(scores).max())) if len(scores) else 1.0
    out, last = [], 0
    for (tok, s, e), sc in zip(toks, scores):
        out.append(html.escape(cleaned[last:s]))
        if abs(sc) >= 0.04:
            a = 0.10 + 0.70 * min(1.0, abs(sc) / ref)
            rgb = "34,197,94" if sc > 0 else "244,63,94"
            out.append(f'<span class="t" title="{sc:+.2f} log-odds" style="background:rgba({rgb},{a:.2f})">'
                       f'{html.escape(tok)}</span>')
        else:
            out.append(html.escape(tok))
        last = e
    out.append(html.escape(cleaned[last:]))
    return '<div class="hl">' + "".join(out) + "</div>"


def make_verdict(p: float, thr: float, band: float) -> dict:
    is_pos = p >= thr
    uncertain = abs(p - thr) < band
    conf = p if is_pos else 1 - p
    if uncertain:
        label, emoji = ("Leaning positive" if is_pos else "Leaning negative"), "🤔"
    else:
        label, emoji = ("Positive" if is_pos else "Negative"), ("😍" if is_pos else "😡")
    level = ("Very confident" if conf >= 0.95 else "Confident" if conf >= 0.8
             else "Moderately confident" if conf >= 0.65 else "Low confidence")
    color = GOLD if uncertain else (POS if is_pos else NEG)
    return {"is_pos": is_pos, "uncertain": uncertain, "label": label, "emoji": emoji,
            "conf": conf, "level": level, "color": color}


def gauge(p: float, thr: float, band: float, color: str) -> go.Figure:
    lo, hi = max(0, (thr - band)) * 100, min(1, (thr + band)) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 40, "color": "#fff"}, "valueformat": ".1f"},
        title={"text": "P(positive)", "font": {"size": 14, "color": MUTED}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"size": 10}},
               "bar": {"color": color, "thickness": 0.30}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, lo], "color": "rgba(244,63,94,.20)"},
                         {"range": [lo, hi], "color": "rgba(245,197,24,.18)"},
                         {"range": [hi, 100], "color": "rgba(34,197,94,.20)"}],
               "threshold": {"line": {"color": "#fff", "width": 3}, "thickness": 0.9, "value": thr * 100}}))
    return style_fig(fig, 235).update_layout(margin=dict(l=24, r=24, t=40, b=0))


def terms_chart(terms: pd.DataFrame, k: int) -> go.Figure:
    t = terms.head(k).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["contribution"], y=t["term"], orientation="h",
        marker_color=[POS if v > 0 else NEG for v in t["contribution"]],
        text=[f"{v:+.2f}" for v in t["contribution"]], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:+.3f} log-odds<extra></extra>"))
    fig.update_layout(title="Terms that moved the prediction (coefficient × TF-IDF)")
    fig.add_vline(x=0, line_color="rgba(255,255,255,.4)", line_width=1)
    return style_fig(fig, max(260, 30 * len(t) + 90))


def waterfall(ex: dict) -> go.Figure:
    vals = [ex["bias"], ex["pos_sum"], ex["neg_sum"], ex["logit"]]
    fig = go.Figure(go.Waterfall(
        x=["Bias", "Positive evidence", "Negative evidence", "Net log-odds"],
        measure=["relative", "relative", "relative", "total"], y=vals,
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
        increasing={"marker": {"color": POS}}, decreasing={"marker": {"color": NEG}},
        totals={"marker": {"color": GOLD}}, connector={"line": {"color": "rgba(255,255,255,.25)"}}))
    fig.update_layout(title=f"Log-odds budget → P(positive) = {pct(sigmoid(ex['logit']))}")
    return style_fig(fig, 340)


def model_card(name: str, p: float | None, src: str, thr: float, band: float) -> str:
    if p is None:
        return h(f"""<div class="mcard"><div class="n">{name}</div>
            <div class="p" style="color:#566074">—</div>
            <div class="d">Not available for this text. Add the fine-tuned weights to enable live inference.</div></div>""")
    v = make_verdict(p, thr, band)
    badge = {"live": "🟢 live", "precomputed": "📦 precomputed (notebook)"}.get(src, src)
    return h(f"""<div class="mcard"><div class="n">{name}</div>
        <div class="p" style="color:{v['color']}">{pct(p)}</div>
        <div class="d">{v['emoji']} {v['label']} · {badge}</div></div>""")


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = np.array([[np.sum((y == 0) & (pred == 0)), np.sum((y == 0) & (pred == 1))],
                   [np.sum((y == 1) & (pred == 0)), np.sum((y == 1) & (pred == 1))]])
    f1s = []
    for c in (0, 1):
        tp = np.sum((y == c) & (pred == c))
        pr = tp / max(1, np.sum(pred == c))
        rc = tp / max(1, np.sum(y == c))
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return {"acc": float(np.mean(y == pred)), "macro_f1": float(np.mean(f1s)), "cm": cm}


# ──────────────────────────────────────────────────────────────────────────────
# Boot
# ──────────────────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.relative_to(ROOT)}`. Re-clone the repo or re-export it from the notebook.")
    st.stop()

B = load_bundle()
META = load_meta()
DEMO = load_demo()
RESULTS = pd.DataFrame(META["results"])
DB_SRC = distil_source()
DB_LIVE = bool(DB_SRC) and distil_libs_ok()

if "review_text" not in st.session_state:
    st.session_state["review_text"] = DEFAULT_TEXT
st.session_state.setdefault("ex_i", 0)


def set_text(t: str) -> None:
    st.session_state["review_text"] = t


def random_text() -> None:
    st.session_state["review_text"] = str(DEMO.sample(1)["text"].iloc[0])


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    md("""<div style="display:flex;align-items:center;gap:10px;margin:2px 0 14px 0">
        <div style="font-size:30px">🎬</div>
        <div><div style="font-weight:800;font-size:1.25rem;line-height:1.1">CineSent</div>
        <div class="muted">IMDB sentiment intelligence</div></div></div>""")
    st.markdown("#### ⚙️ Controls")
    if DB_LIVE:
        model_choice = st.radio("Primary model", [LR_NAME, DB_NAME, ENS_NAME], index=2,
                                help="Decides the headline verdict. Word highlights always explain the Logistic Regression model.")
    else:
        model_choice = LR_NAME
        st.caption("Primary model: **Logistic Regression** (TF-IDF 1-2 grams)")
    THR = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
                    help="P(positive) at or above this value is labelled positive.")
    BAND = st.slider("Uncertainty band (±)", 0.0, 0.30, 0.15, 0.01,
                     help="Predictions this close to the threshold are flagged as low confidence.")
    TOPK = st.slider("Terms to show", 5, 20, 10)
    SHOW_HL = st.toggle("Highlight words in the review", value=True)

    st.markdown("#### 🧩 System status")
    md(f"""<div class="chip g">LR pipeline · {B['size_mb']:.1f} MB · {len(B['names']):,} features</div>""")
    if DB_LIVE:
        md('<div class="chip g">DistilBERT · live</div>')
    elif DB_SRC and not distil_libs_ok():
        md('<div class="chip y">DistilBERT weights found · install torch + transformers</div>')
    else:
        md('<div class="chip b">DistilBERT · precomputed scores only</div>')
        st.caption("Drop the fine-tuned model in `models/distilbert_sentiment/` (see README) to enable live "
                   "DistilBERT and the ensemble.")
    st.markdown("---")
    st.caption("Dataset: [IMDB 50K movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)")
    st.caption("Built with scikit-learn · PyTorch · Streamlit · Plotly")


def predict_all(cleaned: str) -> dict:
    """LR (live) + DistilBERT (live, or precomputed if this exact text is a known notebook review)."""
    t0 = time.perf_counter()
    p_lr = float(B["pipe"].predict_proba([cleaned])[0, 1])
    lat_lr = (time.perf_counter() - t0) * 1000
    p_db, src_db, lat_db = None, "", None
    if DB_LIVE:
        try:
            t0 = time.perf_counter()
            p_db = float(distil_predict([cleaned], DB_SRC)[0])
            lat_db, src_db = (time.perf_counter() - t0) * 1000, "live"
        except Exception as exc:  # degrade gracefully
            st.warning(f"DistilBERT inference failed ({type(exc).__name__}: {exc}). Falling back to Logistic Regression.")
    if p_db is None and cleaned in known_distilbert():
        p_db, src_db = known_distilbert()[cleaned], "precomputed"
    p_ens = (p_lr + p_db) / 2 if p_db is not None else None
    return {"lr": p_lr, "db": p_db, "ens": p_ens, "src_db": src_db, "lat_lr": lat_lr, "lat_db": lat_db}


def primary(pr: dict) -> tuple[float, str]:
    if model_choice == DB_NAME and pr["db"] is not None and pr["src_db"] == "live":
        return pr["db"], DB_NAME
    if model_choice == ENS_NAME and pr["ens"] is not None and pr["src_db"] == "live":
        return pr["ens"], ENS_NAME
    return pr["lr"], LR_NAME


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
best = RESULTS.iloc[0]
spd = load_csv(str(ART / "speed_size.csv"))
md(f"""<div class="hero"><div class="eyebrow">NLP · Sentiment Analysis · Explainable AI</div>
    <h1>CineSent — read the room, one review at a time</h1>
    <p>Classical TF-IDF models versus a fine-tuned DistilBERT on 50,000 IMDB reviews — with honest statistics,
    word-level explanations and a live analyzer you can break on purpose.</p></div>
    <div class="kpis">
    <div class="kpi"><div class="v">{pct(best['Accuracy'], 2)}</div><div class="l">Best test accuracy</div><div class="s">{best['Model']}</div></div>
    <div class="kpi"><div class="v">{META['dataset']['rows_used']:,}</div><div class="l">Unique reviews</div><div class="s">{META['dataset']['duplicates_removed']} duplicates removed</div></div>
    <div class="kpi"><div class="v">{META['tfidf']['n_features']:,}</div><div class="l">TF-IDF features</div><div class="s">unigrams + bigrams</div></div>
    <div class="kpi"><div class="v">{spd.iloc[0]['Reviews/sec']:,.0f}/s</div><div class="l">LR throughput (CPU)</div><div class="s">{spd.iloc[0]['Size (MB)']:.1f} MB model</div></div>
    </div>""")

tabs = st.tabs(["🔮 Live Analyzer", "📦 Batch Lab", "🎞️ Review Explorer", "🏆 Model Arena",
                "🧠 Explainability", "🔬 Errors & Data", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# 1 · LIVE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("### Write or paste a movie review")
        st.text_area("Review text", key="review_text", height=180, max_chars=20000,
                     placeholder="e.g. The cinematography was stunning, but the script fell flat…",
                     label_visibility="collapsed")
        st.caption("Try an edge case — negation, sarcasm, mixed opinions:")
        for row in range(2):
            cols = st.columns(5)
            for col, (lab, txt) in zip(cols, QUICK_EXAMPLES[row * 5:(row + 1) * 5]):
                col.button(lab, key=f"qe_{row}_{lab}", on_click=set_text, args=(txt,), width="stretch")
        st.button("🎲 Random review from the held-out test set", on_click=random_text, width="stretch")

    text = st.session_state["review_text"]
    cleaned = clean_text(text)

    if not cleaned:
        with right:
            md("""<div class="card"><h4>Waiting for a review…</h4>
                <p>Type or paste a review (or tap an example) and the model will score it instantly,
                then show <b>exactly which words</b> pushed it towards positive or negative.</p></div>""")
    else:
        pr = predict_all(cleaned)
        p, who = primary(pr)
        ex = explain(text, B)
        v = make_verdict(p, THR, BAND)
        mixed = min(ex["pos_sum"], -ex["neg_sum"]) >= 0.9 and \
            min(ex["pos_sum"], -ex["neg_sum"]) / max(ex["pos_sum"], -ex["neg_sum"]) >= 0.5

        chips = [f'<span class="chip {"g" if v["is_pos"] else "r"}">{v["level"]} · {pct(v["conf"])}</span>']
        if v["uncertain"]:
            chips.append('<span class="chip y">⚠️ Inside uncertainty band</span>')
        if mixed:
            chips.append('<span class="chip y">⚖️ Mixed signals</span>')
        if ex["n_words"] < 4:
            chips.append('<span class="chip b">Very short text</span>')
        if ex["n_words"] > 190 and who != LR_NAME:
            chips.append('<span class="chip b">Long text — transformer reads first 256 tokens</span>')
        truth = known_truth().get(cleaned)
        if truth is not None:
            ok = (truth == int(v["is_pos"]))
            chips.append(f'<span class="chip {"g" if ok else "r"}">Ground truth: '
                         f'{"positive" if truth else "negative"} {"✓" if ok else "✗"}</span>')
        with right:
            bg = "34,197,94" if v["is_pos"] and not v["uncertain"] else "244,63,94" if not v["is_pos"] and not v["uncertain"] else "245,197,24"
            md(f"""<div class="verdict" style="background:linear-gradient(135deg,rgba({bg},.16),rgba(255,255,255,.02));">
                <div class="big" style="color:{v['color']}">{v['emoji']} {v['label']}</div>
                <div class="sub">{who} · {ex['n_words']} words · {ex['n_known']} known terms ·
                ⚡ {pr['lat_lr']:.1f} ms (LR)</div>{''.join(chips)}</div>""")
            show_fig(gauge(p, THR, BAND, v["color"]), key="gauge")

        st.markdown("#### Model comparison on this review")
        c1, c2, c3 = st.columns(3)
        c1.markdown(model_card(LR_NAME, pr["lr"], "live", THR, BAND), unsafe_allow_html=True)
        c2.markdown(model_card(DB_NAME, pr["db"], pr["src_db"], THR, BAND), unsafe_allow_html=True)
        c3.markdown(model_card(ENS_NAME, pr["ens"], pr["src_db"], THR, BAND), unsafe_allow_html=True)
        if pr["db"] is not None and (pr["lr"] >= THR) != (pr["db"] >= THR):
            st.info("The two models **disagree** on this review — a good candidate for the ensemble (or a human).")

        if SHOW_HL:
            st.markdown("#### Why? Word-level evidence")
            st.markdown(highlight_html(ex), unsafe_allow_html=True)
            md("""<div class="legend"><span>pushes negative</span><div class="bar"></div><span>pushes positive</span>
                <span style="margin-left:auto">hover a word for its log-odds contribution · explains the Logistic Regression model</span></div>""")

        if len(ex["terms"]):
            t1, t2, t3 = st.tabs(["Top terms", "Log-odds budget", "All terms (table)"])
            with t1:
                show_fig(terms_chart(ex["terms"], TOPK), key="terms")
            with t2:
                show_fig(waterfall(ex), key="waterfall")
                st.caption("Logistic Regression is additive: bias + Σ(term weight × TF-IDF) = log-odds, "
                           "and P(positive) = sigmoid(log-odds). Nothing is hidden.")
            with t3:
                st.dataframe(ex["terms"].round(4), width="stretch", hide_index=True)
                st.download_button("⬇️ Download explanation (CSV)", ex["terms"].round(5).to_csv(index=False).encode(),
                                   "explanation.csv", "text/csv")
        else:
            st.warning("None of the words in this text are in the model's vocabulary, so the score is just the bias term.")
        st.caption("⚠️ Models trained on IMDB movie reviews: sarcasm, domain shift (non-movie text) and non-English "
                   "input are known weak spots — see the **Errors & Data** tab.")

# ══════════════════════════════════════════════════════════════════════════════
# 2 · BATCH LAB
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Score many reviews at once")
    st.caption(f"Upload a CSV or paste one review per line (first {MAX_BATCH:,} rows are used). "
               "Add a ground-truth column to get accuracy, macro-F1 and a confusion matrix.")
    mode = st.radio("Input", ["Upload CSV", "Paste reviews"], horizontal=True, key="batch_mode", label_visibility="collapsed")
    df_in, text_col, label_col = None, None, None

    if mode == "Upload CSV":
        up = st.file_uploader("CSV file", type=["csv"], key="batch_file")
        sample = DEMO[["text", "label"]].head(25).assign(label=lambda d: d["label"].map({1: "positive", 0: "negative"}))
        st.download_button("⬇️ Sample CSV (25 test reviews)", sample.to_csv(index=False).encode(), "sample_reviews.csv", "text/csv")
        if up is not None:
            raw = up.getvalue()
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    df_in = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as exc:
                    st.error(f"Could not read this CSV: {exc}")
                    break
            if df_in is not None and len(df_in):
                cols = list(df_in.columns)
                pref = [c for c in cols if c.lower() in ("review", "text", "comment", "content", "sentence", "body")]
                guess = pref[0] if pref else max(cols, key=lambda c: df_in[c].astype(str).str.len().mean())
                c1, c2 = st.columns(2)
                text_col = c1.selectbox("Text column", cols, index=cols.index(guess))
                opts = ["— none —"] + [c for c in cols if c != text_col]
                label_col = c2.selectbox("Ground-truth column (optional)", opts)
                label_col = None if label_col == "— none —" else label_col
            elif df_in is not None:
                st.warning("The file is empty.")
                df_in = None
    else:
        pasted = st.text_area("One review per line", height=180, key="batch_paste",
                              placeholder="This movie was fantastic!\nA total waste of two hours.\nIt was okay, I guess.")
        lines = [l.strip() for l in pasted.splitlines() if l.strip()]
        if lines:
            df_in, text_col = pd.DataFrame({"review": lines}), "review"

    if df_in is not None and text_col is not None:
        n = min(len(df_in), MAX_BATCH)
        if len(df_in) > MAX_BATCH:
            st.warning(f"{len(df_in):,} rows found — scoring the first {MAX_BATCH:,}.")
        if st.button(f"🚀 Analyze {n:,} reviews", type="primary", key="batch_run"):
            sub = df_in.iloc[:n].copy()
            texts = sub[text_col].fillna("").astype(str).tolist()
            cleaned_list = [clean_text(t) for t in texts]
            bar = st.progress(0.0, text="Scoring…")
            t0 = time.perf_counter()
            p_lr_all = B["pipe"].predict_proba(cleaned_list)[:, 1]
            p_all, used = p_lr_all, LR_NAME
            if DB_LIVE and model_choice != LR_NAME:
                try:
                    p_db_all = distil_predict(cleaned_list, DB_SRC, progress=lambda f: bar.progress(f, text="DistilBERT scoring…"))
                    p_all, used = (p_db_all, DB_NAME) if model_choice == DB_NAME else ((p_lr_all + p_db_all) / 2, ENS_NAME)
                except Exception as exc:
                    st.warning(f"DistilBERT failed ({exc}); using Logistic Regression.")
            bar.empty()
            res = pd.DataFrame({"review": texts, "p_positive": p_all})
            res["prediction"] = np.where(res["p_positive"] >= THR, "positive", "negative")
            res["confidence"] = np.where(res["p_positive"] >= THR, res["p_positive"], 1 - res["p_positive"])
            res["uncertain"] = (res["p_positive"] - THR).abs() < BAND
            truth = None
            if label_col:
                m = {"positive": 1, "pos": 1, "1": 1, "true": 1, "negative": 0, "neg": 0, "0": 0, "false": 0}
                truth = sub[label_col].astype(str).str.strip().str.lower().map(m)
                if truth.isna().any():
                    st.warning("Some ground-truth values were not recognised (use positive/negative or 1/0); they are ignored.")
                res["truth"] = truth.map({1: "positive", 0: "negative"}).values
            st.session_state["batch_res"] = {"df": res, "model": used, "secs": time.perf_counter() - t0,
                                             "truth": None if truth is None else truth.values}

    if "batch_res" in st.session_state:
        R = st.session_state["batch_res"]
        res = R["df"]
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reviews scored", f"{len(res):,}")
        k2.metric("Positive share", pct((res['prediction'] == 'positive').mean()))
        k3.metric("Mean confidence", pct(res['confidence'].mean()))
        k4.metric("Low-confidence", f"{int(res['uncertain'].sum()):,}", help="Inside the uncertainty band set in the sidebar")
        st.caption(f"Scored with **{R['model']}** in {R['secs']:.2f}s at threshold {THR:.2f}.")

        g1, g2 = st.columns([1.4, 1])
        with g1:
            fig = go.Figure(go.Histogram(x=res["p_positive"], nbinsx=40, marker_color=GOLD, opacity=.9))
            fig.update_layout(title="Distribution of P(positive)", xaxis_title="P(positive)", yaxis_title="reviews",
                              bargap=0.04)
            fig.add_vline(x=THR, line_dash="dash", line_color="#fff")
            show_fig(style_fig(fig, 320), key="batch_hist")
        with g2:
            cnt = res["prediction"].value_counts()
            fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.62, sort=False,
                                   marker=dict(colors=[POS if l == "positive" else NEG for l in cnt.index]),
                                   textinfo="label+percent"))
            fig.update_layout(title="Predicted sentiment")
            show_fig(style_fig(fig, 320), key="batch_pie")

        if R["truth"] is not None and not pd.isna(R["truth"]).all():
            ok = ~pd.isna(R["truth"])
            y = R["truth"][ok].astype(int)
            pred = (res["p_positive"].values[ok] >= THR).astype(int)
            bm = binary_metrics(y, pred)
            st.markdown("#### Evaluation against your ground truth")
            e1, e2, e3 = st.columns([1, 1, 1.3])
            e1.metric("Accuracy", pct(bm["acc"], 2))
            e2.metric("Macro F1", f"{bm['macro_f1']:.4f}")
            cm = bm["cm"]
            fig = go.Figure(go.Heatmap(z=cm, x=["Pred neg", "Pred pos"], y=["Actual neg", "Actual pos"],
                                       text=cm, texttemplate="%{text}", colorscale=[[0, "#121A2B"], [1, GOLD]],
                                       showscale=False))
            fig.update_layout(title="Confusion matrix", yaxis_autorange="reversed")
            with e3:
                show_fig(style_fig(fig, 260), key="batch_cm")

        view = res.copy()
        view["review"] = view["review"].str.slice(0, 220)
        st.dataframe(view.round(4), hide_index=True, width="stretch", height=380, column_config={
            "p_positive": st.column_config.ProgressColumn("P(positive)", min_value=0.0, max_value=1.0, format="%.3f"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.3f")})
        st.download_button("⬇️ Download full results (CSV)", res.round(5).to_csv(index=False).encode(),
                           "sentiment_predictions.csv", "text/csv", type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# 3 · REVIEW EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Browse 300 held-out test reviews")
    st.caption("None of these were used for training or tuning. Scores for all three models were computed in the "
               "notebook, so this tab works even without the DistilBERT weights.")
    f1, f2, f3 = st.columns([1.3, 1.2, 1])
    judge = f1.radio("Judge by", [LR_NAME, DB_NAME, ENS_NAME], horizontal=True, key="ex_model")
    view_f = f2.selectbox("Show", ["All reviews", "Model mistakes", "Models disagree",
                                   "Confident mistakes (>90%)", "Correct only"], key="ex_view")
    truth_f = f3.selectbox("True sentiment", ["Both", "Positive", "Negative"], key="ex_truth")
    pcol = {LR_NAME: "lr_prob", DB_NAME: "distilbert_prob", ENS_NAME: "ens_prob"}[judge]

    D = DEMO.copy()
    D["pj"] = D[pcol]
    D["pred"] = (D["pj"] >= THR).astype(int)
    D["ok"] = D["pred"] == D["label"]
    D["disagree"] = (D["lr_prob"] >= THR) != (D["distilbert_prob"] >= THR)
    mask = pd.Series(True, index=D.index)
    mask &= {"All reviews": True, "Model mistakes": ~D["ok"], "Models disagree": D["disagree"],
             "Confident mistakes (>90%)": (~D["ok"]) & ((D["pj"] - 0.5).abs() > 0.4),
             "Correct only": D["ok"]}[view_f]
    if truth_f != "Both":
        mask &= D["label"] == (1 if truth_f == "Positive" else 0)
    D = D[mask].reset_index(drop=True)

    if D.empty:
        st.info("No reviews match these filters.")
    else:
        n = len(D)
        st.session_state["ex_i"] = int(min(max(st.session_state["ex_i"], 0), n - 1))

        def nav(step: int | None) -> None:
            st.session_state["ex_i"] = (int(np.random.randint(n)) if step is None
                                        else (st.session_state["ex_i"] + step) % n)

        b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
        b1.button("⬅️ Prev", on_click=nav, args=(-1,), width="stretch", key="ex_prev")
        b2.button("Next ➡️", on_click=nav, args=(1,), width="stretch", key="ex_next")
        b3.button("🎲 Random", on_click=nav, args=(None,), width="stretch", key="ex_rand")
        b4.markdown(f"<div class='muted' style='padding-top:8px'>Review <b>{st.session_state['ex_i'] + 1}</b> of <b>{n}</b> "
                    f"in this view · {int(D['ok'].sum())} correct / {int((~D['ok']).sum())} wrong for {judge}</div>",
                    unsafe_allow_html=True)

        r = D.iloc[st.session_state["ex_i"]]
        truth_lbl = "positive" if r["label"] == 1 else "negative"
        tcol = POS if r["label"] == 1 else NEG
        okchip = '<span class="chip g">✓ Correct</span>' if r["ok"] else '<span class="chip r">✗ Wrong</span>'
        md(f"""<div class="card"><span class="chip" style="color:{tcol};border-color:{tcol}">Ground truth: {truth_lbl}</span>
            {okchip}<span class="chip">{int(r['words'])} words</span>
            {'<span class="chip y">Models disagree</span>' if r['disagree'] else ''}</div>""")

        ex = explain(r["text"], B)
        if SHOW_HL:
            st.markdown(highlight_html(ex), unsafe_allow_html=True)
            md("""<div class="legend"><span>pushes negative</span><div class="bar"></div><span>pushes positive</span>
                <span style="margin-left:auto">Logistic Regression explanation</span></div>""")
        else:
            st.write(r["text"])
        c1, c2, c3 = st.columns(3)
        c1.markdown(model_card(LR_NAME, float(r["lr_prob"]), "precomputed", THR, BAND), unsafe_allow_html=True)
        c2.markdown(model_card(DB_NAME, float(r["distilbert_prob"]), "precomputed", THR, BAND), unsafe_allow_html=True)
        c3.markdown(model_card(ENS_NAME, float(r["ens_prob"]), "precomputed", THR, BAND), unsafe_allow_html=True)
        if len(ex["terms"]):
            show_fig(terms_chart(ex["terms"], TOPK), key="ex_terms")

        with st.expander(f"📋 All {n} reviews in this view"):
            tbl = D[["label", "words", "lr_prob", "distilbert_prob", "ens_prob", "text"]].copy()
            tbl["label"] = tbl["label"].map({1: "positive", 0: "negative"})
            tbl["text"] = tbl["text"].str.slice(0, 200)
            st.dataframe(tbl.round(4), hide_index=True, width="stretch", height=360, column_config={
                "lr_prob": st.column_config.ProgressColumn("LR", min_value=0.0, max_value=1.0, format="%.2f"),
                "distilbert_prob": st.column_config.ProgressColumn("DistilBERT", min_value=0.0, max_value=1.0, format="%.2f"),
                "ens_prob": st.column_config.ProgressColumn("Ensemble", min_value=0.0, max_value=1.0, format="%.2f")})

# ══════════════════════════════════════════════════════════════════════════════
# 4 · MODEL ARENA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Honest model comparison (5,000-review test set, evaluated once)")
    metric = st.radio("Metric", ["Accuracy", "Macro F1", "ROC-AUC", "Log-loss"], horizontal=True, key="arena_metric",
                      help="Log-loss: lower is better. Accuracy shows 95% bootstrap confidence intervals.")
    R_ = RESULTS[RESULTS["Model"] != "Dummy"].copy()
    asc = metric == "Log-loss"
    # horizontal bars draw the first row at the bottom, so put the best model last
    R_ = R_.sort_values(metric, ascending=not asc)
    colors = [GOLD if m == best["Model"] and metric != "Log-loss" else BLUE for m in R_["Model"]]
    if metric == "Log-loss":
        colors = [GOLD if v == R_[metric].min() else BLUE for v in R_[metric]]
    err = dict(type="data", symmetric=False, array=(R_["CI high"] - R_["Accuracy"]),
               arrayminus=(R_["Accuracy"] - R_["CI low"]), color="rgba(255,255,255,.6)") if metric == "Accuracy" else None
    fig = go.Figure(go.Bar(x=R_[metric], y=R_["Model"], orientation="h", marker_color=colors, error_x=err,
                           text=[f"{v:.4f}" for v in R_[metric]], textposition="inside", insidetextanchor="start",
                           hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:.4f}<extra></extra>"))
    lo_x = float(R_[metric].min())
    fig.update_xaxes(range=[max(0, lo_x - (0.03 if metric != "Log-loss" else 0.05)),
                            float(R_["CI high"].max() + 0.01) if metric == "Accuracy" else float(R_[metric].max() + 0.03)])
    fig.update_layout(title=f"Test {metric}" + (" with 95% bootstrap CI" if metric == "Accuracy" else ""))
    show_fig(style_fig(fig, 340), key="arena_bar")

    a1, a2, a3, a4 = st.columns(4)
    ens_r = RESULTS[RESULTS["Model"].str.startswith("Ensemble")].iloc[0]
    db_r = RESULTS[RESULTS["Model"] == "DistilBERT"].iloc[0]
    lr_r = RESULTS[RESULTS["Model"] == "Logistic Regression"].iloc[0]
    a1.metric("Ensemble vs DistilBERT", f"{100 * (ens_r['Accuracy'] - db_r['Accuracy']):+.2f} pts", "p < 0.0001")
    a2.metric("DistilBERT vs LR", f"{100 * (db_r['Accuracy'] - lr_r['Accuracy']):+.2f} pts", "p = 0.37 (n.s.)", delta_color="off")
    uni, bi = META["unigram_vs_bigram_val_acc"]
    a3.metric("Bigrams help (val)", f"{100 * (bi - uni):+.2f} pts", f"{pct(uni, 2)} → {pct(bi, 2)}")
    a4.metric("LR vs DistilBERT speed", f"{spd.iloc[0]['Reviews/sec'] / spd.iloc[1]['Reviews/sec']:.1f}× faster",
              f"{spd.iloc[1]['Size (MB)'] / spd.iloc[0]['Size (MB)']:.0f}× smaller", delta_color="off")

    st.markdown("#### Full results")
    tbl = RESULTS.copy()
    tbl["95% CI"] = tbl.apply(lambda r: f"{r['CI low']:.3f} – {r['CI high']:.3f}", axis=1)
    st.dataframe(tbl[["Model", "Accuracy", "95% CI", "Macro F1", "ROC-AUC", "Log-loss"]], hide_index=True, width="stretch",
                 column_config={"Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.4, max_value=1.0, format="%.4f"),
                                "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
                                "Macro F1": st.column_config.NumberColumn(format="%.4f"),
                                "Log-loss": st.column_config.NumberColumn(format="%.4f")})

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("#### Paired significance (exact McNemar)")
        pt = pd.DataFrame(PAIRED_TESTS, columns=["Comparison", "Only 1st right", "Only 2nd right", "p-value", "Verdict"])
        st.dataframe(pt, hide_index=True, width="stretch")
        st.caption("A 0.4-point accuracy gap between DistilBERT and LR is within noise on 5,000 reviews. Averaging the two "
                   "is the only improvement that is statistically solid — their mistakes are different.")
    with s2:
        st.markdown("#### Speed vs size")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["LR (CPU)", "DistilBERT (GPU fp16)"], y=spd["Reviews/sec"], name="Reviews / sec",
                             marker_color=GOLD, text=[f"{v:,.0f}" for v in spd["Reviews/sec"]], textposition="outside"))
        fig.add_trace(go.Bar(x=["LR (CPU)", "DistilBERT (GPU fp16)"], y=spd["Size (MB)"], name="Size (MB)",
                             marker_color=VIOLET, text=[f"{v:,.1f}" for v in spd["Size (MB)"]], textposition="outside"))
        fig.update_layout(barmode="group", yaxis_type="log", title="Throughput and model size (log scale)")
        show_fig(style_fig(fig, 300, legend=True), key="arena_speed")

    st.markdown("#### Diagnostics from the notebook")
    d1, d2, d3 = st.tabs(["ROC & comparison", "Calibration", "DistilBERT training"])
    d1.image(str(ASSETS / "model_comparison.png"), width="stretch")
    d2.image(str(ASSETS / "calibration.png"), width=520)
    d2.caption("Logistic Regression tracks the diagonal closely, so its percentages behave like real confidence. "
               "DistilBERT is a little over-confident in the mid-range (e.g. predictions near 0.84 are right ~72% of the time).")
    d3.image(str(ASSETS / "distilbert_training.png"), width="stretch")
    st.caption(f"Tuned on validation only — NB α={META['best_params']['naive_bayes_alpha']}, "
               f"SVM C={META['best_params']['linear_svm_C']}, LR C={META['best_params']['logreg_C']}. "
               f"DistilBERT: {META['transformer']['epochs']} epochs, lr {META['transformer']['lr']}, max_len "
               f"{META['transformer']['max_len']}, {META['transformer']['params'] / 1e6:.0f}M parameters, "
               f"{META['transformer']['train_seconds'] / 60:.1f} min on a Kaggle T4.")

# ══════════════════════════════════════════════════════════════════════════════
# 5 · EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### What did the model actually learn?")
    st.caption("Logistic Regression assigns one weight per word / word-pair. Positive weight → pushes towards a positive review.")
    top = load_csv(str(ART / "lr_top_terms.csv")).sort_values("coefficient")
    N = st.slider("Terms per side", 5, 25, 15, key="expl_n")
    neg_t, pos_t = top.head(N), top.tail(N)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pushes POSITIVE", "Pushes NEGATIVE"), horizontal_spacing=0.14)
    pos_s, neg_s = pos_t.sort_values("coefficient"), neg_t.sort_values("coefficient", ascending=False)
    fig.add_trace(go.Bar(x=pos_s["coefficient"], y=pos_s["term"], orientation="h", marker_color=POS,
                         hovertemplate="<b>%{y}</b><br>weight %{x:+.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=neg_s["coefficient"], y=neg_s["term"], orientation="h", marker_color=NEG,
                         hovertemplate="<b>%{y}</b><br>weight %{x:+.2f}<extra></extra>"), row=1, col=2)
    fig.update_layout(title="Strongest words and phrases (LR coefficients)")
    show_fig(style_fig(fig, max(380, 26 * N + 120)), key="expl_top")

    st.markdown("#### 🔎 Term lookup")
    q = st.text_input("Type a word or two-word phrase", value="not good", key="expl_q",
                      placeholder="e.g. boring, not bad, waste of").strip().lower()
    if q:
        j = B["vocab"].get(q)
        l1, l2 = st.columns([1, 2], gap="large")
        with l1:
            if j is None:
                st.warning(f"“{q}” is not in the 200k-term vocabulary (too rare, or longer than 2 words).")
            else:
                w = B["coef"][j]
                rank = int((B["coef"] > w).sum()) + 1
                st.metric("Weight", f"{w:+.3f}", "pushes positive" if w > 0 else "pushes negative", delta_color="normal" if w > 0 else "inverse")
                st.caption(f"Rank {rank:,} of {len(B['coef']):,} (1 = most positive).")
        with l2:
            hit = np.where(np.char.find(B["padded"], f" {q} ") >= 0)[0]
            if len(hit):
                sub = pd.DataFrame({"term": B["names"][hit], "weight": B["coef"][hit]})
                sub = sub.reindex(sub["weight"].abs().sort_values(ascending=False).index).head(12).round(3)
                st.markdown(f"**Terms containing “{q}”**")
                st.dataframe(sub, hide_index=True, width="stretch", height=300, column_config={
                    "weight": st.column_config.ProgressColumn("weight", min_value=-16.0, max_value=16.0, format="%.2f")})
            else:
                st.caption("No other vocabulary term contains this phrase.")
        st.caption("💡 Compare *good*, *not good*, *bad* and *not bad* — bigrams are how a bag-of-words model handles negation.")

    st.markdown("#### 🧩 Strongest two-word phrases")
    bi_idx = np.where(B["is_bigram"])[0]
    order = bi_idx[np.argsort(B["coef"][bi_idx])]
    cA, cB = st.columns(2)
    cA.markdown("**Most negative**")
    cA.dataframe(pd.DataFrame({"phrase": B["names"][order[:12]], "weight": B["coef"][order[:12]].round(2)}),
                 hide_index=True, width="stretch", height=300)
    cB.markdown("**Most positive**")
    cB.dataframe(pd.DataFrame({"phrase": B["names"][order[::-1][:12]], "weight": B["coef"][order[::-1][:12]].round(2)}),
                 hide_index=True, width="stretch", height=300)

    with st.expander("🖼️ Static figures from the notebook"):
        st.image(str(ASSETS / "lr_top_terms.png"), width="stretch")
        st.image(str(ASSETS / "lr_local_explanations.png"), width="stretch",
                 caption="Local explanations: a correct positive, a correct negative and the most confident mistake.")

# ══════════════════════════════════════════════════════════════════════════════
# 6 · ERRORS & DATA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    e1, e2, e3 = st.tabs(["Where it fails", "Edge cases", "Data & EDA"])
    with e1:
        st.markdown(f"### Error analysis — featured model: {META['featured_model']}")
        st.caption("The featured model was chosen on the validation set (not on test).")
        abl = load_csv(str(ART / "accuracy_by_length.csv"))
        x1, x2 = st.columns([1, 1.2], gap="large")
        with x1:
            fig = go.Figure(go.Bar(x=abl["bucket"], y=abl["accuracy"] * 100, marker_color=BLUE,
                                   text=[f"{a * 100:.1f}%<br>n={n:,}" for a, n in zip(abl["accuracy"], abl["reviews"])],
                                   textposition="outside", cliponaxis=False))
            fig.update_yaxes(range=[85, 100], title="accuracy (%)")
            fig.update_xaxes(title="review length (words)")
            fig.update_layout(title="Accuracy by review length")
            show_fig(style_fig(fig, 360), key="err_len")
        with x2:
            st.image(str(ASSETS / "error_analysis.png"), width="stretch")
        st.markdown(
            "- Very short reviews (≤100 words) are easiest; long reviews mix praise and criticism and are harder.\n"
            "- **40.7%** of test reviews exceed DistilBERT's 256-token window, so it only reads their first part.\n"
            "- Both models are wrong together on **203 of 5,000** reviews (4.1%) — often ambiguous text, plot summaries "
            "with little opinion, or wording that contradicts the label.")
    with e2:
        st.markdown("### Stress-testing with hand-written edge cases")
        ce = load_csv(str(ART / "custom_examples.csv")).copy()
        ce["challenge"] = EDGE_TAGS[:len(ce)]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ce["review"], x=ce["LR P(pos)"], orientation="h", name=LR_NAME, marker_color=GOLD))
        fig.add_trace(go.Bar(y=ce["review"], x=ce["DistilBERT P(pos)"], orientation="h", name=DB_NAME, marker_color=VIOLET))
        fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(255,255,255,.5)")
        fig.update_layout(barmode="group", title="P(positive) on tricky sentences", xaxis=dict(range=[0, 1]))
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
        show_fig(style_fig(fig, 640, legend=True).update_layout(margin=dict(l=10, r=10, t=60, b=10)), key="err_edge")
        st.dataframe(ce[["challenge", "review", "LR P(pos)", "DistilBERT P(pos)"]], hide_index=True, width="stretch", column_config={
            "LR P(pos)": st.column_config.ProgressColumn("LR P(pos)", min_value=0.0, max_value=1.0, format="%.3f"),
            "DistilBERT P(pos)": st.column_config.ProgressColumn("DistilBERT P(pos)", min_value=0.0, max_value=1.0, format="%.3f")})
        st.caption("Clear sentiment, negation (“not good”) and reversals are handled well by both. They stumble on "
                   "contrast (“predictable, yet I couldn't stop watching”), neutral text (LR calls “nothing special, nothing "
                   "terrible” strongly negative) and LR also misses “not bad at all”. The sarcasm example is only caught because "
                   "of the word “boring”. Paste any of these in the Live Analyzer to see why.")
    with e3:
        ds = META["dataset"]
        st.markdown("### Dataset & preprocessing")
        c1, c2 = st.columns([1, 1.3], gap="large")
        with c1:
            fig = go.Figure(go.Pie(labels=["Train", "Validation", "Test"], values=[ds["train"], ds["val"], ds["test"]], hole=.6,
                                   marker=dict(colors=[BLUE, VIOLET, GOLD]), sort=False, textinfo="label+value"))
            fig.update_layout(title="Stratified split")
            show_fig(style_fig(fig, 320), key="data_split")
        with c2:
            md(f"""<div class="card"><h4>Cleaning</h4><p>{ds['rows_raw']:,} raw reviews → <b>{ds['duplicates_removed']}</b> exact duplicates
                removed → <b>{ds['rows_used']:,}</b> unique reviews. HTML entities and <code>&lt;br /&gt;</code> tags stripped.
                Splits are stratified and fixed by seed; models fit on train, hyper-parameters chosen on validation, and
                the test set is touched <b>once</b>.</p></div>
                <div class="card"><h4>Features</h4><p>TF-IDF with 1-2 grams, <code>min_df=3</code>, <code>max_df=0.9</code>,
                sublinear TF, up to 200k features. Adding bigrams lifted validation accuracy from
                {pct(META['unigram_vs_bigram_val_acc'][0], 2)} to {pct(META['unigram_vs_bigram_val_acc'][1], 2)}.</p></div>""")
        st.image(str(ASSETS / "eda_overview.png"), width="stretch")
        st.image(str(ASSETS / "eda_top_words.png"), width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# 7 · ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### How it works")
    md("""<div class="flow">
        <div class="step"><b>1 · Clean</b><span>HTML unescape, strip tags, collapse whitespace, drop duplicates</span></div>
        <div class="step"><b>2 · Split</b><span>Stratified 39,582 / 5,000 / 5,000 with a fixed seed</span></div>
        <div class="step"><b>3 · Vectorise</b><span>TF-IDF 1-2 grams, 200k features, sublinear TF</span></div>
        <div class="step"><b>4 · Model</b><span>Naive Bayes, Linear SVM, Logistic Regression, fine-tuned DistilBERT</span></div>
        <div class="step"><b>5 · Evaluate</b><span>Accuracy + 95% bootstrap CI, F1, AUC, log-loss, calibration, McNemar</span></div>
        <div class="step"><b>6 · Explain</b><span>Additive log-odds decomposition per word and bigram</span></div></div>""")
    a, b = st.columns(2, gap="large")
    with a:
        md("""<div class="card"><h4>Why Logistic Regression is the live model</h4>
            <p>It is within noise of DistilBERT on accuracy (p = 0.37), 3× faster on a CPU, 67× smaller,
            well calibrated and fully explainable. DistilBERT earns its keep as part of an <b>ensemble</b>,
            which is the best model on the test set.</p></div>
            <div class="card"><h4>Run it locally</h4>
            <p><code>pip install -r requirements.txt</code><br><code>streamlit run app.py</code></p>
            <p class="muted">Optional live DistilBERT: <code>pip install -r requirements-transformer.txt</code> and place the
            fine-tuned folder in <code>models/distilbert_sentiment/</code> (or set <code>DISTILBERT_PATH</code> to a folder or Hugging Face repo id).</p></div>""")
    with b:
        md("""<div class="card"><h4>Known limitations</h4><ul>
            <li>Trained on English movie reviews only — expect weaker results on other domains, languages or emoji-heavy text.</li>
            <li>Bag-of-words cannot truly read sarcasm, long-range negation or neutral text; it scores the words it sees.</li>
            <li>DistilBERT reads at most 256 tokens; LR reads the whole review.</li>
            <li>Labels are binary — “mixed” reviews are forced to one side (the app flags them).</li></ul></div>
            <div class="card"><h4>Stack</h4><p>Python · scikit-learn · PyTorch + 🤗 Transformers (training) · Streamlit · Plotly · Pandas</p></div>""")
    st.caption("Model facts shown in this app come from `artifacts/model_meta.json`, exported by the notebook.")

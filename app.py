"""
CineSent design polish for app.py: static, CSS-first, no animations.

Usage:  python patch_design.py            # patches ./app.py, keeps app.py.design.bak
        python patch_design.py path/to/app.py

Works on the original app.py and on one already patched with patch_animations.py.
Each edit is an exact-match replace; if an anchor is missing or appears more than
once, the script aborts and writes nothing.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "app.py")
src = path.read_text(encoding="utf-8")
MARK = "/* ═══════════ DESIGN POLISH ═══════════ */"
if MARK in src:
    sys.exit("Design polish already applied - nothing to do.")

DESIGN_CSS = MARK + r'''
:root{--gold:#F5C518;--glass:rgba(255,255,255,.04);--edge:rgba(255,255,255,.09);--ink2:#B7C0D3;}
::selection{background:rgba(245,197,24,.35);color:#fff;}
html{scroll-behavior:smooth;}
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.14);border-radius:99px;border:2px solid transparent;background-clip:padding-box;}
::-webkit-scrollbar-thumb:hover{background:rgba(245,197,24,.5);background-clip:padding-box;}
header[data-testid="stHeader"]{background:transparent;}

/* typography */
h1,h2,h3,h4{font-feature-settings:"ss01","cv11";}
h3{font-size:1.5rem;padding-bottom:.15rem;}
h4{color:#E8ECF4;}
.stCaption,[data-testid="stCaptionContainer"]{color:#8B95A9;}
hr{border:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.16),transparent);margin:1.2rem 0;}

/* hero: grid texture, glass edge, badges */
.hero{background:
  linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px) 0 0/34px 34px,
  linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px) 0 0/34px 34px,
  linear-gradient(135deg,rgba(245,197,24,.16),rgba(96,165,250,.09) 55%,rgba(167,139,250,.14));
 border:1px solid rgba(245,197,24,.22);
 box-shadow:0 24px 70px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.12);}
.hero h1{font-size:2.8rem;letter-spacing:-.025em;}
.hero .eyebrow{display:inline-block;padding:4px 12px;border-radius:99px;background:rgba(245,197,24,.12);
 border:1px solid rgba(245,197,24,.35);margin-bottom:6px;}
.hero .badges{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.hero .badges span{padding:5px 12px;border-radius:99px;font-size:.78rem;font-weight:600;color:#E8ECF4;
 background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14);backdrop-filter:blur(6px);}

/* glass cards with a gold accent line */
.kpi,.card,.mcard,.step{background:linear-gradient(160deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
 border:1px solid var(--edge);box-shadow:0 10px 30px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.07);}
.kpi{position:relative;overflow:hidden;padding:18px 20px;}
.kpi:before{content:"";position:absolute;left:0;top:0;height:3px;width:100%;
 background:linear-gradient(90deg,var(--gold),rgba(245,197,24,0));}
.kpi .v{font-size:1.9rem;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
.kpi .s{display:inline-block;margin-top:6px;padding:2px 9px;border-radius:99px;background:rgba(245,197,24,.10);}
.mcard .p{font-size:1.9rem;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
.card h4{color:#fff;}
.step b{display:inline-block;padding:2px 9px;border-radius:99px;background:rgba(245,197,24,.12);}

/* verdict + chips */
.verdict{border-left:4px solid var(--c,#F5C518);box-shadow:0 16px 44px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.08);}
.verdict .big{letter-spacing:-.02em;}
.chip{backdrop-filter:blur(4px);letter-spacing:.01em;}

/* highlighted review */
.hl{box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 8px 26px rgba(0,0,0,.25);}
.hl .t{box-shadow:0 0 0 1px rgba(255,255,255,.06);}

/* tabs as a segmented pill bar */
div[data-baseweb="tab-list"]{gap:4px;padding:5px;border-radius:14px;background:rgba(255,255,255,.04);
 border:1px solid var(--edge);overflow-x:auto;}
button[data-baseweb="tab"]{border-radius:10px;color:var(--ink2);}
button[data-baseweb="tab"][aria-selected="true"]{background:rgba(245,197,24,.12);color:#F5C518;
 box-shadow:inset 0 0 0 1px rgba(245,197,24,.35);}
div[data-baseweb="tab-highlight"],div[data-baseweb="tab-border"]{display:none;}

/* buttons */
.stButton>button,.stDownloadButton>button{background:rgba(255,255,255,.05);color:#E8ECF4;min-height:2.6rem;}
button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#F5C518,#E0A800)!important;
 color:#1A1400!important;border:0!important;font-weight:800;box-shadow:0 8px 24px rgba(245,197,24,.28);}
button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.08);color:#1A1400!important;}

/* inputs */
.stTextArea textarea,.stTextInput input,div[data-baseweb="select"]>div{background:rgba(255,255,255,.04)!important;
 border:1px solid var(--edge)!important;border-radius:14px!important;}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:rgba(245,197,24,.6)!important;
 box-shadow:0 0 0 3px rgba(245,197,24,.18)!important;}
div[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.035);border:1.5px dashed rgba(245,197,24,.4);border-radius:16px;}

/* metrics, charts, tables as glass panels */
div[data-testid="stMetric"]{padding:14px 16px;border-radius:16px;background:linear-gradient(160deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
 border:1px solid var(--edge);box-shadow:0 8px 24px rgba(0,0,0,.25);}
div[data-testid="stMetricValue"]{font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
div[data-testid="stPlotlyChart"]{padding:8px 6px;border-radius:18px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);}
div[data-testid="stDataFrame"]{border:1px solid var(--edge);box-shadow:0 8px 24px rgba(0,0,0,.25);}
div[data-testid="stExpander"]{border:1px solid var(--edge);border-radius:16px;background:rgba(255,255,255,.03);}
div[data-testid="stAlert"]{border-radius:14px;border:1px solid var(--edge);}

/* sidebar */
section[data-testid="stSidebar"]{box-shadow:6px 0 40px rgba(0,0,0,.35);}
section[data-testid="stSidebar"] h4{font-size:.8rem;text-transform:uppercase;letter-spacing:.14em;color:#F5C518;margin-top:1.1rem;}
section[data-testid="stSidebar"] .chip{display:block;width:fit-content;}
'''

BADGES = ('<div class="badges"><span>🧠 TF-IDF + Logistic Regression</span><span>🤖 DistilBERT</span>'
          '<span>🔍 Word-level explanations</span><span>📊 Bootstrap CIs &amp; McNemar</span></div>')

EDITS = [
    ('</style>\n"""\nst.markdown(CSS, unsafe_allow_html=True)',
     DESIGN_CSS + '</style>\n"""\nst.markdown(CSS, unsafe_allow_html=True)'),
    ('live analyzer you can break on purpose.</p>',
     'live analyzer you can break on purpose.</p>' + BADGES),
]

out = src
for i, (old, new) in enumerate(EDITS, 1):
    c = out.count(old)
    if c != 1:
        sys.exit(f"Edit {i} aborted: anchor found {c} times (expected 1). Nothing was written.")
    out = out.replace(old, new)

path.with_suffix(".py.design.bak").write_text(src, encoding="utf-8")
path.write_text(out, encoding="utf-8")
print(f"Design polish applied to {path} ({len(EDITS)} edits). Backup: {path.with_suffix('.py.design.bak')}")

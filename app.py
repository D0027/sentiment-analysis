"""
Applies the CineSent animation upgrade to app.py (UI-only changes).

Usage:  python patch_animations.py            # patches ./app.py, keeps app.py.bak
        python patch_animations.py path/to/app.py

Every edit is an exact-match replace and the script aborts without writing
anything if an anchor is missing or appears more than once, so the rest of
the file is guaranteed untouched.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "app.py")
src = path.read_text(encoding="utf-8")

if "/* ═══════════ ANIMATIONS ═══════════ */" in src:
    sys.exit("Already patched - nothing to do.")

ANIM_CSS = r'''
/* ═══════════ ANIMATIONS ═══════════ */
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes popIn{0%{opacity:0;transform:scale(.88) translateY(10px)}60%{opacity:1;transform:scale(1.03)}100%{opacity:1;transform:none}}
@keyframes shimmer{to{background-position:200% center}}
@keyframes floaty{0%,100%{transform:rotate(-10deg) translateY(0)}50%{transform:rotate(-4deg) translateY(-14px)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes drift{from{transform:translate3d(-3%,-2%,0) scale(1)}to{transform:translate3d(3%,3%,0) scale(1.12)}}
@keyframes filmroll{to{background-position:28px 0}}
@keyframes sweep{to{left:170%}}
@keyframes tokIn{from{opacity:0;transform:translateY(6px) scale(.85)}to{opacity:1;transform:none}}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 transparent}50%{box-shadow:0 0 38px 2px color-mix(in srgb,var(--c,#F5C518) 35%,transparent)}}
@keyframes pulseRing{0%,100%{box-shadow:0 0 0 0 rgba(245,197,24,.35)}50%{box-shadow:0 0 0 9px rgba(245,197,24,0)}}
@keyframes pulseDot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.5)}}

/* drifting colour blobs behind everything */
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
 background:radial-gradient(420px 420px at 18% 28%,rgba(245,197,24,.07),transparent 70%),
 radial-gradient(520px 520px at 82% 62%,rgba(96,165,250,.07),transparent 70%),
 radial-gradient(460px 460px at 55% 95%,rgba(167,139,250,.07),transparent 70%);
 animation:drift 22s ease-in-out infinite alternate;}
.block-container{animation:fadeUp .7s cubic-bezier(.2,.8,.2,1) both;}

/* hero: rotating aurora, shimmering title, floating clapperboard, scrolling film strip */
.hero:before{content:"";position:absolute;inset:-60%;z-index:0;pointer-events:none;
 background:conic-gradient(from 0deg,rgba(245,197,24,.20),rgba(96,165,250,.14),rgba(167,139,250,.20),rgba(245,197,24,.20));
 filter:blur(70px);animation:spin 22s linear infinite;}
.hero>*{position:relative;z-index:1;}
.hero:after{animation:floaty 6s ease-in-out infinite;}
.hero h1{background:linear-gradient(90deg,#fff,#F5C518,#FDE68A,#fff);background-size:200% auto;
 -webkit-background-clip:text;background-clip:text;color:transparent;animation:shimmer 5s linear infinite;}
.hero .eyebrow{animation:fadeUp .6s .1s both;}
.hero p{animation:fadeUp .6s .3s both;}
.hero .strip{height:12px;margin-top:20px;border-radius:4px;opacity:.55;
 background:repeating-linear-gradient(90deg,rgba(255,255,255,.18) 0 14px,transparent 14px 28px);
 animation:filmroll 1.6s linear infinite;}

/* cards: staggered entrance + hover lift */
.kpi,.card,.mcard,.step{animation:fadeUp .7s cubic-bezier(.2,.8,.2,1) both;
 transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease;}
.kpi:nth-child(2),.step:nth-child(2){animation-delay:.08s}
.kpi:nth-child(3),.step:nth-child(3){animation-delay:.16s}
.kpi:nth-child(4),.step:nth-child(4){animation-delay:.24s}
.step:nth-child(5){animation-delay:.32s}.step:nth-child(6){animation-delay:.40s}
.kpi:hover,.card:hover,.mcard:hover,.step:hover{transform:translateY(-5px);
 border-color:rgba(245,197,24,.5);box-shadow:0 14px 34px rgba(245,197,24,.12);}

/* verdict: bouncy pop-in + breathing glow in the verdict colour */
.verdict{animation:popIn .6s cubic-bezier(.2,.9,.3,1.2) both,glow 2.8s ease-in-out .7s infinite;}
.verdict .big{animation:fadeUp .6s .15s both;}
.chip{animation:popIn .45s both;}
.chip:nth-of-type(2){animation-delay:.10s}.chip:nth-of-type(3){animation-delay:.20s}
.chip:nth-of-type(4){animation-delay:.30s}.chip:nth-of-type(5){animation-delay:.40s}
section[data-testid="stSidebar"] .chip.g:before{content:"";display:inline-block;width:7px;height:7px;
 border-radius:50%;background:#22C55E;margin-right:7px;animation:pulseDot 1.6s ease-in-out infinite;}

/* word highlights light up one after another */
.hl .t{display:inline-block;animation:tokIn .5s both;transition:transform .15s ease,filter .15s ease;}
.hl .t:hover{transform:scale(1.18);filter:brightness(1.35);}

/* tabs, buttons, inputs */
div[data-baseweb="tab-panel"]{animation:fadeUp .5s both;}
button[data-baseweb="tab"]{transition:color .2s ease,transform .2s ease;}
button[data-baseweb="tab"]:hover{transform:translateY(-2px);color:#FDE68A;}
.stButton>button,.stDownloadButton>button{position:relative;overflow:hidden;}
.stButton>button:before,.stDownloadButton>button:before{content:"";position:absolute;top:0;left:-120%;width:55%;height:100%;
 background:linear-gradient(100deg,transparent,rgba(255,255,255,.2),transparent);transform:skewX(-20deg);}
.stButton>button:hover:before,.stDownloadButton>button:hover:before{animation:sweep .8s ease;}
.stButton>button:active{transform:scale(.97);}
button[data-testid="stBaseButton-primary"]{animation:pulseRing 2.4s ease-in-out infinite;}
.stTextArea textarea{transition:box-shadow .25s ease,border-color .25s ease;}
.stTextArea textarea:focus{box-shadow:0 0 0 3px rgba(245,197,24,.25),0 0 30px rgba(245,197,24,.15);}

/* charts, tables, images, metrics fade in */
div[data-testid="stPlotlyChart"],div[data-testid="stDataFrame"],
div[data-testid="stMetric"],div[data-testid="stImage"] img{animation:fadeUp .8s cubic-bezier(.2,.8,.2,1) both;}

@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;}}
'''

EDITS = [
    # 1 · animation CSS, just before the closing </style>
    ('</style>\n"""\nst.markdown(CSS, unsafe_allow_html=True)',
     ANIM_CSS + '</style>\n"""\nst.markdown(CSS, unsafe_allow_html=True)'),
    # 2 · film strip in the hero
    ('word-level explanations and a live analyzer you can break on purpose.</p></div>',
     'word-level explanations and a live analyzer you can break on purpose.</p><div class="strip"></div></div>'),
    # 3a · verdict glow colour
    ('md(f"""<div class="verdict" style="background:linear-gradient(135deg,rgba({bg},.16),rgba(255,255,255,.02));">',
     'md(f"""<div class="verdict" style="--c:{v[\'color\']};background:linear-gradient(135deg,rgba({bg},.16),rgba(255,255,255,.02));">'),
    # 3b · staggered word highlights
    ('    for (tok, s, e), sc in zip(toks, scores):',
     '    for n, ((tok, s, e), sc) in enumerate(zip(toks, scores)):'),
    ('''            out.append(f'<span class="t" title="{sc:+.2f} log-odds" style="background:rgba({rgb},{a:.2f})">\'''',
     '''            out.append(f'<span class="t" title="{sc:+.2f} log-odds" '
                       f'style="background:rgba({rgb},{a:.2f});animation-delay:{min(n * 35, 1800)}ms">\''''),
    # 4 · toast after the batch finishes
    ('''                                             "truth": None if truth is None else truth.values}\n''',
     '''                                             "truth": None if truth is None else truth.values}\n'''
     '''            st.toast(f"Scored {len(texts):,} reviews with {used}", icon="🚀")\n'''),
]

out = src
for i, (old, new) in enumerate(EDITS, 1):
    c = out.count(old)
    if c != 1:
        sys.exit(f"Edit {i} aborted: anchor found {c} times (expected 1). Nothing was written.")
    out = out.replace(old, new)

path.with_suffix(".py.bak").write_text(src, encoding="utf-8")
path.write_text(out, encoding="utf-8")
print(f"Patched {path} ({len(EDITS)} edits). Backup: {path.with_suffix('.py.bak')}")
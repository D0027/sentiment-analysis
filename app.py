"""
CineSent motion v2: safe animation layer for the ORIGINAL app.py.

Usage:  python patch_motion_v2.py            # patches ./app.py, keeps app.py.motion.bak
        python patch_motion_v2.py path/to/app.py

Safety rules used here (the v1 patch broke them):
  * no full-screen fixed layers, no transforms on Streamlit's own containers
  * only elements this app creates are animated (hero, KPIs, cards, verdict, chips, word highlights)
  * animations never leave an element hidden if they fail to run
  * respects prefers-reduced-motion
Every edit is an exact-match replace; if an anchor is missing or repeated, nothing is written.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "app.py")
src = path.read_text(encoding="utf-8")
if "MOTION v2" in src:
    sys.exit("Motion v2 already applied - nothing to do.")
if "/* ═══════════ ANIMATIONS ═══════════ */" in src:
    sys.exit("The old v1 animation patch is in this file. Restore the working app.py first "
             "(cp app.py.bak app.py, or git checkout <good-commit> -- app.py), then run this again.")

MOTION_CSS = r'''
/* ═══════════ MOTION v2 ═══════════ */
@keyframes m-rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes m-pop{0%{opacity:0;transform:scale(.9) translateY(8px)}65%{opacity:1;transform:scale(1.025)}100%{opacity:1;transform:none}}
@keyframes m-aurora{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes m-shimmer{to{background-position:200% center}}
@keyframes m-float{0%,100%{transform:rotate(-10deg) translateY(0)}50%{transform:rotate(-5deg) translateY(-12px)}}
@keyframes m-film{to{background-position:28px 0}}
@keyframes m-glow{0%,100%{box-shadow:0 0 0 0 rgba(0,0,0,0)}50%{box-shadow:0 0 34px 2px color-mix(in srgb,var(--c,#F5C518) 32%,transparent)}}
@keyframes m-ring{0%,100%{box-shadow:0 0 0 0 rgba(245,197,24,.35)}50%{box-shadow:0 0 0 9px rgba(245,197,24,0)}}
@keyframes m-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.5)}}
@keyframes m-token{from{opacity:0;transform:translateY(5px) scale(.88)}to{opacity:1;transform:none}}
@keyframes m-sweep{to{left:170%}}

/* hero: slow aurora, shimmering title, floating clapperboard, scrolling film strip */
.hero{background-size:220% 220%;animation:m-aurora 16s ease-in-out infinite;}
.hero:after{animation:m-float 6s ease-in-out infinite;}
.hero h1{background-size:220% auto;animation:m-shimmer 6s linear infinite;
 background-image:linear-gradient(90deg,#fff,#F5C518,#FDE68A,#fff);}
.hero .eyebrow{animation:m-rise .6s .05s ease-out backwards;}
.hero h1{animation:m-shimmer 6s linear infinite,m-rise .7s .15s ease-out backwards;}
.hero p{animation:m-rise .6s .3s ease-out backwards;}
.hero .strip{height:12px;margin-top:20px;border-radius:4px;opacity:.5;
 background:repeating-linear-gradient(90deg,rgba(255,255,255,.2) 0 14px,transparent 14px 28px);
 animation:m-film 1.6s linear infinite;}

/* KPI cards: staggered entrance, lift on hover */
.kpi,.card,.mcard,.step{transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease;}
.kpi{animation:m-rise .65s ease-out backwards;}
.kpi:nth-child(1){animation-delay:.35s}.kpi:nth-child(2){animation-delay:.43s}
.kpi:nth-child(3){animation-delay:.51s}.kpi:nth-child(4){animation-delay:.59s}
.step{animation:m-rise .6s ease-out backwards;}
.step:nth-child(2){animation-delay:.06s}.step:nth-child(3){animation-delay:.12s}
.step:nth-child(4){animation-delay:.18s}.step:nth-child(5){animation-delay:.24s}.step:nth-child(6){animation-delay:.30s}
.card,.mcard{animation:m-rise .55s ease-out backwards;}
.kpi:hover,.card:hover,.mcard:hover,.step:hover{transform:translateY(-4px);
 border-color:rgba(245,197,24,.5);box-shadow:0 14px 32px rgba(245,197,24,.12);}

/* verdict: bouncy pop-in then a slow glow in its own colour; chips pop one by one */
.verdict{animation:m-pop .6s cubic-bezier(.2,.9,.3,1.2) backwards,m-glow 3s ease-in-out .8s infinite;}
.verdict .big{animation:m-rise .5s .12s ease-out backwards;}
.chip{animation:m-pop .45s ease-out backwards;}
.chip:nth-of-type(2){animation-delay:.10s}.chip:nth-of-type(3){animation-delay:.20s}
.chip:nth-of-type(4){animation-delay:.30s}.chip:nth-of-type(5){animation-delay:.40s}
section[data-testid="stSidebar"] .chip.g:before{content:"";display:inline-block;width:7px;height:7px;
 border-radius:50%;background:#22C55E;margin-right:7px;animation:m-dot 1.6s ease-in-out infinite;}

/* word highlights light up one after another */
.hl .t{display:inline-block;animation:m-token .45s ease-out backwards;transition:transform .15s ease,filter .15s ease;}
.hl .t:hover{transform:scale(1.15);filter:brightness(1.3);}

/* buttons: shine sweep on hover, pulsing primary */
.stButton>button,.stDownloadButton>button{position:relative;overflow:hidden;}
.stButton>button:before,.stDownloadButton>button:before{content:"";position:absolute;top:0;left:-120%;width:55%;height:100%;
 background:linear-gradient(100deg,transparent,rgba(255,255,255,.2),transparent);transform:skewX(-20deg);pointer-events:none;}
.stButton>button:hover:before,.stDownloadButton>button:hover:before{animation:m-sweep .8s ease;}
.stButton>button:active{transform:scale(.97);}
button[data-testid="stBaseButton-primary"]{animation:m-ring 2.4s ease-in-out infinite;}
.stTextArea textarea{transition:box-shadow .25s ease,border-color .25s ease;}
.stTextArea textarea:focus{box-shadow:0 0 0 3px rgba(245,197,24,.25),0 0 28px rgba(245,197,24,.14);}
button[data-baseweb="tab"]{transition:color .2s ease;}
button[data-baseweb="tab"]:hover{color:#FDE68A;}

@media (prefers-reduced-motion:reduce){.hero,.hero *,.kpi,.card,.mcard,.step,.verdict,.verdict *,.chip,.hl .t,
 .stButton>button,.stButton>button:before{animation:none!important;transition:none!important;}}
'''

EDITS = [
    # 1 · motion CSS, just before the closing </style>
    ('</style>\n"""\nst.markdown(CSS, unsafe_allow_html=True)',
     MOTION_CSS + '</style>\n"""\nst.markdown(CSS, unsafe_allow_html=True)'),
    # 2 · film strip at the bottom of the hero
    ('</div>\n    <div class="kpis">',
     '<div class="strip"></div></div>\n    <div class="kpis">'),
    # 3a · verdict glow colour
    ('md(f"""<div class="verdict" style="background:linear-gradient(135deg,rgba({bg},.16),rgba(255,255,255,.02));">',
     'md(f"""<div class="verdict" style="--c:{v[\'color\']};background:linear-gradient(135deg,rgba({bg},.16),rgba(255,255,255,.02));">'),
    # 3b · staggered word highlights
    ('    for (tok, s, e), sc in zip(toks, scores):',
     '    for n, ((tok, s, e), sc) in enumerate(zip(toks, scores)):'),
    ('''            out.append(f'<span class="t" title="{sc:+.2f} log-odds" style="background:rgba({rgb},{a:.2f})">\'''',
     '''            out.append(f'<span class="t" title="{sc:+.2f} log-odds" '
                       f'style="background:rgba({rgb},{a:.2f});animation-delay:{min(n * 30, 1500)}ms">\''''),
    # 4 · toast when the batch finishes
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

path.with_suffix(".py.motion.bak").write_text(src, encoding="utf-8")
path.write_text(out, encoding="utf-8")
print(f"Motion v2 applied to {path} ({len(EDITS)} edits). Backup: {path.with_suffix('.py.motion.bak')}")
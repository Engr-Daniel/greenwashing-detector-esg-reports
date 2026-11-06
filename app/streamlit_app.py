"""Greenwashing detector for ESG / sustainability reports.  Run: streamlit run app/streamlit_app.py

UI only - all logic lives in src/greenwash.
"""
import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from greenwash.analyse import MIX, analyse_pdf, analyse_sentences, claim_mask, claim_mix, models, scorecard, signal_counts  # noqa: E402
from greenwash.paths import METRICS, PROCESSED, SAMPLES  # noqa: E402
from greenwash.pdftext import sentences  # noqa: E402
from greenwash.scoring import explain, grade, load_signals  # noqa: E402

# ----------------------------------------------------------------------------- design tokens
INK, INK2, INK3, BORDER, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e7e5e0", "#ecebe7", "#ffffff"
BLUE, ORANGE, AQUA, NAVY, FOREST = "#2a78d6", "#eb6834", "#1baf7a", "#0d366b", "#0f5132"
SERIES = [BLUE, ORANGE, AQUA]
REPO = "https://github.com/Engr-Daniel/greenwashing-detector-esg-reports"
GRADE_TXT = {"A": "Specific & verifiable", "B": "Mostly specific", "C": "Mixed", "D": "Mostly vague", "E": "Vague"}
SIG = {s.name: s for s in load_signals()}
DISCLAIMER = ("Scores measure how **specific and verifiable** the language of a report is. They are not a finding "
              "that any company is greenwashing, and they don't check whether the claims are true.")

st.set_page_config(page_title="Greenwashing Detector · ESG reports", page_icon="🔎", layout="wide",
                   initial_sidebar_state="auto")
st.markdown(f"""
<style>
  .block-container {{padding-top:1.4rem; padding-bottom:3rem; max-width:1400px;}}
  [data-testid="stSidebar"] {{border-right:1px solid {BORDER};}}
  .sb-h {{font-size:0.7rem; font-weight:700; letter-spacing:0.08em; color:{INK3}; text-transform:uppercase; margin:16px 0 4px;}}
  .hero {{background:linear-gradient(120deg, #0b3d2e 0%, {FOREST} 55%, #187a52 100%); color:#fff; border-radius:18px;
          padding:26px 30px 22px; margin-bottom:18px; position:relative; overflow:hidden;}}
  .hero:after {{content:""; position:absolute; right:-70px; top:-90px; width:320px; height:320px; border-radius:50%;
          background:radial-gradient(circle, rgba(170,240,200,0.35) 0%, rgba(170,240,200,0) 68%);}}
  .hero .eyebrow {{font-size:0.74rem; letter-spacing:0.12em; text-transform:uppercase; opacity:0.85; font-weight:600;}}
  .hero h1 {{color:#fff; font-size:2.0rem; line-height:1.2; margin:6px 0 8px; padding:0; font-weight:700;}}
  .hero p {{color:rgba(255,255,255,0.88); margin:0; font-size:1.0rem; max-width:880px; line-height:1.5;}}
  .hero .meta {{display:flex; flex-wrap:wrap; gap:8px; margin-top:16px;}}
  .hero .chip {{font-size:0.76rem; padding:4px 11px; border-radius:999px; background:rgba(255,255,255,0.12);
          border:1px solid rgba(255,255,255,0.24); color:#fff;}}
  .hero .chip.warn {{background:rgba(255,214,102,0.18); border-color:rgba(255,214,102,0.5);}}
  .kpis {{display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:12px; margin-bottom:6px;}}
  .kpi {{background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; padding:16px 18px 14px; min-height:122px;
         box-shadow:0 1px 2px rgba(16,24,40,0.04);}}
  .kpi .label {{font-size:0.7rem; color:{INK2}; text-transform:uppercase; letter-spacing:0.07em; font-weight:700;}}
  .kpi .value {{font-size:1.8rem; font-weight:700; color:{INK}; line-height:1.15; margin-top:8px; font-variant-numeric:tabular-nums;}}
  .kpi .unit {{font-size:0.9rem; font-weight:500; color:{INK2}; margin-left:2px;}}
  .kpi .sub {{font-size:0.8rem; color:{INK2}; margin-top:6px; line-height:1.35;}}
  .gradebox {{display:flex; gap:18px; align-items:center; background:{SURFACE}; border:1px solid {BORDER};
         border-radius:14px; padding:16px 20px; margin-bottom:14px;}}
  .gradebox .g {{flex:0 0 74px; height:74px; border-radius:50%; background:{NAVY}; color:#fff; font-size:2.3rem;
         font-weight:800; display:flex; align-items:center; justify-content:center;}}
  .gradebox .t {{font-size:1.05rem; color:{INK}; line-height:1.5;}}
  .gradebox .s {{font-size:0.86rem; color:{INK2}; margin-top:2px;}}
  .card {{background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; padding:16px 18px; height:100%;
          box-shadow:0 1px 2px rgba(16,24,40,0.04);}}
  .card h4 {{margin:0 0 10px; padding:0; font-size:0.98rem;}}
  .muted {{color:{INK2}; font-size:0.84rem; line-height:1.5;}}
  .quote {{border-left:4px solid {BORDER}; padding:6px 12px; margin:8px 0; font-size:0.9rem; color:{INK}; line-height:1.45;}}
  .quote.bad {{border-left-color:{ORANGE};}} .quote.good {{border-left-color:{AQUA};}}
  .quote .m {{font-size:0.74rem; color:{INK3}; margin-top:3px;}}
  .tag {{display:inline-block; font-size:0.72rem; padding:2px 8px; border-radius:999px; margin:2px 4px 2px 0;
         border:1px solid {BORDER}; color:{INK2}; background:#faf9f6;}}
  .tag.red {{border-color:rgba(235,104,52,0.45); color:#9a3b12; background:rgba(235,104,52,0.07);}}
  .tag.green {{border-color:rgba(27,175,122,0.45); color:#0f6b4b; background:rgba(27,175,122,0.07);}}
  .msg {{display:flex; gap:12px; align-items:flex-start; margin-bottom:12px;}}
  .msg .n {{flex:0 0 26px; height:26px; border-radius:50%; background:{FOREST}; color:#fff; font-weight:700;
         font-size:0.8rem; display:flex; align-items:center; justify-content:center;}}
  .msg .t {{font-weight:700; font-size:0.92rem; color:{INK};}} .msg .b {{font-size:0.85rem; color:{INK2}; line-height:1.45;}}
  .sec {{font-size:1.18rem; font-weight:700; color:{INK}; margin:22px 0 2px;}}
  .sec-sub {{font-size:0.88rem; color:{INK2}; margin:0 0 12px;}}
  .notice {{font-size:0.84rem; color:{INK2}; background:#fbf7ec; border:1px solid #efe3c2; border-radius:10px;
         padding:10px 14px; margin:4px 0 14px;}}
  .foot {{border-top:1px solid {BORDER}; margin-top:34px; padding-top:14px; font-size:0.78rem; color:{INK3}; line-height:1.6;}}
  .foot a {{color:{INK2};}}
  div[data-testid="stTabs"] [data-baseweb="tab-list"] {{gap:6px; border-bottom:1px solid {BORDER};}}
  div[data-testid="stTabs"] button {{padding:10px 14px;}}
  div[data-testid="stTabs"] button p {{font-size:0.95rem; font-weight:600;}}
  .mobile-hint {{display:none;}}
  @media (max-width: 768px) {{
    .block-container {{padding-left:0.8rem; padding-right:0.8rem; padding-top:3.2rem;}}
    .hero {{padding:18px 18px 16px; border-radius:14px;}} .hero h1 {{font-size:1.4rem;}} .hero p {{font-size:0.9rem;}}
    .hero:after {{width:180px; height:180px; right:-50px; top:-60px;}}
    .kpis {{grid-template-columns:repeat(2, 1fr); gap:8px;}}
    .kpi {{min-height:0; padding:12px 12px 10px;}} .kpi .value {{font-size:1.4rem;}} .kpi .sub {{font-size:0.74rem;}}
    .gradebox {{padding:12px 14px; gap:12px;}} .gradebox .g {{flex-basis:56px; height:56px; font-size:1.8rem;}}
    .gradebox .t {{font-size:0.95rem;}}
    .mobile-hint {{display:block; font-size:0.78rem; margin-top:12px; opacity:0.9;}}
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {{overflow-x:auto; flex-wrap:nowrap;}}
    div[data-testid="stTabs"] button {{padding:8px 8px; white-space:nowrap;}}
    div[data-testid="stTabs"] button p {{font-size:0.85rem;}}
  }}
</style>""", unsafe_allow_html=True)
try:
    MOBILE = any(t in (st.context.headers.get("User-Agent") or "") for t in ("Mobi", "Android", "iPhone"))
except Exception:  # noqa: BLE001
    MOBILE = False


# ----------------------------------------------------------------------------- helpers
def kpi_row(items):
    st.markdown('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="label">{lab}</div><div class="value">{val}<span class="unit">{unit}</span></div>'
        f'<div class="sub">{sub}</div></div>' for lab, val, unit, sub in items) + "</div>", unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="sec-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def style(fig, title, sub=None, height=340, legend=False):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{INK2};font-weight:400'>{sub}</span>"
                                             if sub else ""), x=0, xanchor="left", y=0.97, font=dict(size=15, color=INK)),
        height=height, margin=dict(l=6, r=24, t=74 if sub else 48, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=12), barcornerradius=4,
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font=dict(color=INK, size=12)),
        showlegend=legend, legend=(dict(orientation="h", y=-0.12, yanchor="top", x=0, title=None) if MOBILE else
                                   dict(orientation="h", y=1.0, yanchor="bottom", x=0, title=None)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER)
    return fig


CFG = {"displayModeBar": False}


def tags(names):
    return "".join(f'<span class="tag {SIG[n].kind}">{html.escape(SIG[n].label)}</span>' for n in names if n in SIG)


def pct(x):
    return "–" if x is None or x != x else f"{x:.0%}"


# ----------------------------------------------------------------------------- data
@st.cache_data
def static():
    return json.loads((PROCESSED / "case_studies.json").read_text()), json.loads(METRICS.read_text())


@st.cache_resource(show_spinner="Loading models…")
def warm():
    return models()


@st.cache_data(show_spinner="Reading the report and scoring every sentence…", max_entries=8)
def run_pdf(digest: str, data: bytes):
    df, n = analyse_pdf(data)
    return df, n


@st.cache_data(show_spinner="Scoring…", max_entries=16)
def run_text(text: str):
    return analyse_sentences([(1, s) for s in sentences(text)])


CASES, METR = static()
warm()
BENCH = CASES["benchmark"]

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🔎 Report to analyse")
    src = st.radio("Source", ["Nigerian case studies", "Upload a PDF", "Paste text", "Fictional demo report"],
                   label_visibility="collapsed")
    case_key, df, n_pages, meta = None, None, None, {}
    if src == "Nigerian case studies":
        keys = list(CASES["reports"])
        case_key = st.selectbox("Report", keys, format_func=lambda k: f'{CASES["reports"][k]["company"]} '
                                                                      f'({CASES["reports"][k]["year"]})')
        meta = CASES["reports"][case_key]
        st.caption("Pre-computed from the public PDFs. Upload the PDF to see every claim.")
    elif src == "Upload a PDF":
        up = st.file_uploader("Sustainability / ESG / annual report (PDF)", type=["pdf"])
        name = st.text_input("Company & year (optional)", placeholder="e.g. MTN Nigeria 2024")
        if up is not None:
            data = up.getvalue()
            df, n_pages = run_pdf(hashlib.sha1(data).hexdigest(), data)
            meta = {"company": name or up.name.rsplit(".", 1)[0], "year": "", "title": up.name}
    elif src == "Paste text":
        txt = st.text_area("Paste report text", height=220, placeholder="Paste a few paragraphs or a whole report…")
        name = st.text_input("Company & year (optional)")
        if txt.strip():
            df, n_pages = run_text(txt), None
            meta = {"company": name or "Pasted text", "year": "", "title": "Pasted text"}
    else:
        txt = (SAMPLES / "fictional_example_report.txt").read_text()
        df, n_pages = run_text(txt), None
        meta = {"company": "Savanna Agro-Foods Plc (fictional)", "year": 2025, "title": "Fictional demo report"}
        st.caption("A made-up company. The text mixes strong and weak claims on purpose.")
    if df is not None:
        st.markdown('<div class="sb-h">Detection</div>', unsafe_allow_html=True)
        th = st.slider("Claim-detector threshold", 0.3, 0.9, float(getattr(models()[0], "threshold_", 0.55)), 0.05,
                       help="Higher = only sentences the model is very confident are environmental claims.")
        df = df.copy()
        df["is_claim"] = claim_mask(df, th)
    st.divider()
    st.link_button("Code & methodology on GitHub", REPO, width="stretch")
    st.caption("Built by **Daniel Oyewale** · Data & Sustainability, Petgrave.io")

# ----------------------------------------------------------------------------- current report summary
if case_key:
    sc = CASES["reports"][case_key]["scorecard"]
    sigs = pd.DataFrame(CASES["reports"][case_key]["signals"])
    hist = CASES["reports"][case_key]["spec_hist"]
    claims = None
    targets = pd.DataFrame(CASES["reports"][case_key]["targets"])
elif df is not None and len(df):
    sc = scorecard(df, n_pages)
    sigs = signal_counts(df)
    c = df[df.is_claim]
    hist = c.specificity.value_counts(bins=list(range(0, 105, 5)), sort=False).astype(int).tolist()
    claims = c.reset_index(drop=True)
    targets = pd.DataFrame([dict(page=r.page, sentence=r.sentence, **r.target) for r in c[c.target.notna()].itertuples()])
else:
    sc = None

label = f'{meta.get("company", "")} {meta.get("year", "")}'.strip() if meta else ""
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">ESG disclosure analytics · Greenwashing risk</div>
  <h1>Are these climate claims specific enough to verify?</h1>
  <p>Reads a sustainability report, finds every environmental claim with a model trained on 2,600 expert-labelled
     sentences, then scores each one for specificity and verifiability against the EU and UK green-claims rules:
     vague wording, absolute claims, reliance on offsets, missing baselines, missing numbers and missing verification.</p>
  <div class="meta">
    <span class="chip">Report: <b>{html.escape(label) or "none selected"}</b></span>
    <span class="chip">Claim detector macro-F1 {METR["claim_detector"]["macro_f1"]:.2f}</span>
    <span class="chip warn">Measures disclosure quality, not proof of greenwashing</span>
  </div>
  <div class="mobile-hint">Tap <b>›</b> at the top left to choose or upload a report.</div>
</div>""", unsafe_allow_html=True)

t_score, t_claims, t_targets, t_cmp, t_meth = st.tabs(["Scorecard", "Claims", "Targets", "Compare reports",
                                                        "Method & model"])

if sc is None:
    with t_score:
        st.info("Upload a PDF or paste text in the sidebar to start." if src != "Nigerian case studies" else "")
    for t in (t_claims, t_targets):
        with t:
            st.info("No report loaded yet.")
else:
    n_claims = sc["claims"]
    # ======================================================================== SCORECARD
    with t_score:
        g = sc["report_grade"]
        top_red = sigs[(sigs.kind == "red")].sort_values("claims", ascending=False).head(2)
        red_txt = " and ".join(f"<b>{r.signal.lower()}</b> ({r.share:.0%})" for r in top_red.itertuples() if r.claims)
        st.markdown(f"""<div class="gradebox"><div class="g">{g}</div><div><div class="t"><b>{GRADE_TXT[g]}.</b>
            {html.escape(label)} makes <b>{n_claims}</b> environmental claims. Their median specificity is
            <b>{sc['median_specificity']:.0f}/100</b> (international benchmark median {BENCH['median']:.0f}).
            {f'The most common red flags are {red_txt}.' if red_txt else ''}</div>
            <div class="s">Report score {sc['report_score']:.0f}/100 = 75% average claim specificity + 25% target completeness.</div>
            </div></div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="notice">⚖️ {DISCLAIMER.replace("**", "<b>", 1).replace("**", "</b>", 1)}</div>',
                    unsafe_allow_html=True)
        kpi_row([
            ("Environmental claims", f"{n_claims:,}", "", f"{sc['claim_density']:.0%} of {sc['sentences']:,} sentences"
             + (f" · {sc['pages']} pages" if sc.get("pages") else "")),
            ("Vague wording", pct(sc["share_vague"]), "", "claims using generic green terms"),
            ("Quantified", pct(sc["share_quantified"]), "", "claims with a number and unit"),
            ("Externally verified", pct(sc["share_verified"]), "", "mention assurance, certification or SBTi"),
            ("Measurable targets", f"{sc['targets']}", "", f"{sc['targets_with_baseline']} with a baseline year · "
                                                          f"{sc['targets_complete']} complete"),
        ])
        st.write("")
        a, b = st.columns(2, gap="large")
        with a:
            s2 = sigs.copy()
            s2["order"] = s2.kind.map({"red": 0, "green": 1})
            s2 = s2.sort_values(["order", "share"], ascending=[False, True])
            fig = go.Figure()
            for kind, col, name in (("red", ORANGE, "Red flag"), ("green", AQUA, "Evidence")):
                d = s2[s2.kind == kind]
                fig.add_trace(go.Bar(y=d.signal, x=d.share, orientation="h", marker_color=col, name=name,
                                     text=[f"{v:.0%}" for v in d.share], textposition="outside", cliponaxis=False,
                                     textfont=dict(color=INK2, size=11),
                                     hovertemplate="%{y}: %{x:.0%} of claims<extra></extra>"))
            style(fig, "What the claims contain", "Share of environmental claims with each signal", height=430, legend=True)
            fig.update_xaxes(tickformat=".0%", range=[0, max(0.1, s2.share.max() * 1.25)])
            fig.update_yaxes(gridcolor="rgba(0,0,0,0)", categoryorder="array", categoryarray=list(s2.signal))
            st.plotly_chart(fig, width="stretch", config=CFG)
        with b:
            bins = [f"{i}–{i+5}" for i in range(0, 100, 5)]
            tot, btot = max(sum(hist), 1), max(sum(BENCH["hist"]), 1)
            fig = go.Figure([
                go.Bar(x=bins, y=[h / tot for h in hist], name=label[:28] or "This report", marker_color=BLUE,
                       hovertemplate="%{x}: %{y:.0%}<extra></extra>"),
                go.Bar(x=bins, y=[h / btot for h in BENCH["hist"]], name="International benchmark", marker_color="#b9b7b0",
                       hovertemplate="%{x}: %{y:.0%}<extra></extra>")])
            style(fig, "Specificity of claims vs benchmark",
                  "0 = vague, 100 = specific" if MOBILE else
                  f"0 = vague, 100 = specific & verifiable · benchmark: {BENCH['n']} expert-labelled claims by listed firms",
                  height=430, legend=True)
            fig.update_layout(bargap=0.15, bargroupgap=0.05)
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
            st.plotly_chart(fig, width="stretch", config=CFG)

        # strongest / weakest quotes
        if case_key:
            weak, strong = CASES["reports"][case_key]["weakest"], CASES["reports"][case_key]["strongest"]
        else:
            cc = claims[claims.p_claim >= 0.6] if (claims.p_claim >= 0.6).sum() >= 3 else claims
            weak = cc.sort_values(["specificity", "gw_risk"], ascending=[True, False]).head(5).to_dict("records")
            strong = claims.sort_values(["specificity", "gw_risk"], ascending=[False, True]).head(5).to_dict("records")

        def quotes(rows, kind):
            return "".join(
                f'<div class="quote {kind}">“{html.escape(r["sentence"])}”<div class="m">p. {r["page"]} · specificity '
                f'{r["specificity"]}/100 · {tags(list(r["red_flags"]) + list(r["evidence"]))}</div></div>' for r in rows)
        a, b = st.columns(2, gap="large")
        a.markdown(f'<div class="card"><h4>Least specific claims</h4>{quotes(weak[:5], "bad")}</div>', unsafe_allow_html=True)
        b.markdown(f'<div class="card"><h4>Most specific claims</h4>{quotes(strong[:5], "good")}</div>', unsafe_allow_html=True)

        # questions for the company
        qs = []
        if (sc["share_vague"] or 0) > 0.15:
            qs.append(("Define the green terms", "What evidence backs terms like “sustainable” and “eco-friendly”, and "
                                                 "what do they cover (product, site or group)?"))
        if sc["targets_with_baseline"] < sc["targets"] or sc["targets"] == 0:
            qs.append(("State baselines and interim milestones", "Which base year and boundary (Scope 1, 2, 3) does "
                                                                 "each target use, and what are the interim milestones?"))
        if (sc["share_verified"] or 0) < 0.05:
            qs.append(("Get the numbers assured", "Are emissions and other environmental KPIs independently assured "
                                                  "(ISAE 3410 / ISO 14064-3), and to what level?"))
        if sigs.set_index("signal").loc[SIG["offset_reliance"].label, "claims"] > 0:
            qs.append(("Separate reductions from offsets", "How much of any neutrality claim relies on carbon credits, "
                                                           "and of what type and vintage?"))
        qs = qs[:4] or [("Keep it up", "Claims are mostly specific. Check that the numbers reconcile with the data tables.")]
        st.markdown('<div class="card" style="margin-top:16px"><h4>Questions an investor or regulator should ask</h4>' +
                    "".join(f'<div class="msg"><div class="n">{i}</div><div><div class="t">{t}</div><div class="b">{b_}'
                            f'</div></div></div>' for i, (t, b_) in enumerate(qs, 1)) + "</div>", unsafe_allow_html=True)

    # ======================================================================== CLAIMS
    with t_claims:
        if claims is None:
            section("Quoted excerpts", "Case-study results store only short excerpts. Upload the PDF in the sidebar to "
                                       "score every claim.")
            ex = pd.DataFrame(CASES["reports"][case_key]["weakest"] + CASES["reports"][case_key]["strongest"])
            st.dataframe(ex.assign(red_flags=ex.red_flags.map(lambda r: ", ".join(SIG[x].label for x in r)),
                                   evidence=ex.evidence.map(lambda r: ", ".join(SIG[x].label for x in r)))
                         [["page", "specificity", "sentence", "red_flags", "evidence"]],
                         hide_index=True, width="stretch",
                         column_config={"specificity": st.column_config.ProgressColumn("Specificity", min_value=0,
                                                                                       max_value=100, format="%d")})
        else:
            section(f"All {len(claims):,} environmental claims", "Filter, then select a row to see why it scored the way "
                                                                 "it did.")
            f1, f2 = st.columns([2, 1])
            flt = f1.multiselect("Show claims with red flag", [SIG[k].label for k in SIG if SIG[k].kind == "red"])
            rng = f2.slider("Specificity range", 0, 100, (0, 100))
            view = claims[claims.specificity.between(*rng)]
            if flt:
                want = {k for k in SIG if SIG[k].label in flt}
                view = view[view.red_flags.map(lambda r: bool(want & set(r)))]
            tbl = view.assign(
                flags=view.red_flags.map(lambda r: ", ".join(SIG[x].label for x in r)),
                ev=view.evidence.map(lambda r: ", ".join(SIG[x].label for x in r)),
                risk=view.gw_risk * 100)[["page", "specificity", "grade", "sentence", "flags", "ev", "risk"]]
            ev = st.dataframe(tbl, hide_index=True, width="stretch", height=420, on_select="rerun",
                              selection_mode="single-row", key="claims_tbl", column_config={
                                  "page": st.column_config.NumberColumn("Page", width="small"),
                                  "specificity": st.column_config.ProgressColumn("Specificity", min_value=0, max_value=100,
                                                                                 format="%d"),
                                  "grade": st.column_config.TextColumn("Grade", width="small"),
                                  "sentence": st.column_config.TextColumn("Claim", width="large"),
                                  "flags": "Red flags", "ev": "Evidence",
                                  "risk": st.column_config.ProgressColumn("Language risk", min_value=0, max_value=100,
                                                                          format="%d%%", help="Probability from the "
                                                                          "greenwashing-language model (CV AUC 0.78)")})
            st.download_button("⬇ Download all claims (CSV)", claims.drop(columns=["signals", "target"]).assign(
                red_flags=claims.red_flags.map("; ".join), evidence=claims.evidence.map("; ".join)).to_csv(index=False)
                .encode(), "claims.csv", "text/csv")
            sel = getattr(getattr(ev, "selection", None), "rows", None)
            if len(view):
                r = view.iloc[sel[0] if sel else 0]
                section("Why this claim scored " + f"{r.specificity}/100")
                st.markdown(f'<div class="quote">“{html.escape(r.sentence)}”<div class="m">page {r.page} · '
                            f'claim probability {r.p_claim:.0%} · language risk {r.gw_risk:.0%}</div></div>',
                            unsafe_allow_html=True)
                rows = explain(r.signals)
                if rows:
                    st.dataframe(pd.DataFrame(rows)[["signal", "points", "evidence", "why", "reference"]].rename(
                        columns=str.capitalize), hide_index=True, width="stretch")
                else:
                    st.caption("No red flags or evidence signals: a neutral statement (score 50).")
                with st.expander("Optional: ask an LLM for a second opinion (your own Anthropic API key)"):
                    key = st.text_input("Anthropic API key", type="password", help="Used once for this request; "
                                                                                  "not stored.")
                    model = st.text_input("Model", "claude-sonnet-5-5")
                    if st.button("Review this claim", disabled=not key):
                        try:
                            import anthropic
                            msg = anthropic.Anthropic(api_key=key).messages.create(
                                model=model, max_tokens=500, messages=[{"role": "user", "content": (
                                    "You are an ESG disclosure analyst. Assess this corporate environmental claim against "
                                    "the EU Empowering Consumers Directive and the UK CMA Green Claims Code. In under 150 "
                                    "words: (1) verdict: specific / partly specific / vague; (2) what information is "
                                    "missing to verify it; (3) one question to ask the company. Don't assume facts.\n\n"
                                    f"Claim: {r.sentence}")}])
                            st.markdown(msg.content[0].text)
                        except Exception as e:  # noqa: BLE001
                            st.error(f"LLM request failed: {e}")

    # ======================================================================== TARGETS
    with t_targets:
        section("Measurable targets", "Forward-looking targets with a year, or a target word plus a percentage. "
                                      "General “commitment to sustainability” wording is not counted.")
        if targets is None or targets.empty:
            st.info("No measurable environmental targets found. That's a finding in itself: pledges without a year or "
                    "number can't be tracked.")
        else:
            tt = targets.copy()
            chk = lambda v: "✓" if v is not None and v == v else "✗"  # noqa: E731
            tt["Year"] = tt.target_year.map(lambda v: f"{int(v)}" if v == v and v is not None else "✗")
            tt["Baseline"] = tt.baseline_year.map(lambda v: f"{int(v)}" if v == v and v is not None else "✗")
            tt["% change"] = tt.reduction_pct.map(lambda v: f"{v:g}%" if v == v and v is not None else "✗")
            tt["Scope"] = tt.scope.map(lambda v: v if isinstance(v, str) else "✗")
            st.dataframe(tt[["page", "type", "Year", "Baseline", "% change", "Scope", "completeness", "sentence"]],
                         hide_index=True, width="stretch", column_config={
                             "page": st.column_config.NumberColumn("Page", width="small"), "type": "Target type",
                             "completeness": st.column_config.ProgressColumn("Completeness", min_value=0, max_value=1,
                                                                             format="percent"),
                             "sentence": st.column_config.TextColumn("Target text", width="large")})
            st.caption("Completeness = share of the four elements a verifiable target needs (target year, baseline "
                       "year, quantified change, emissions scope), per the GHG Protocol and IFRS S2 para 33.")

# ======================================================================== COMPARE
with t_cmp:
    section("Nigerian sustainability reports compared", "Same model and rules applied to three public reports from "
                                                        "companies listed on the Nigerian Exchange.")
    rows = []
    for k, r in CASES["reports"].items():
        s = r["scorecard"]
        rows.append({"Report": f'{r["company"]} ({r["year"]})', "Sector": r["sector"], "Grade": s["report_grade"],
                     "Score": s["report_score"], "Claims": s["claims"], "Median specificity": s["median_specificity"],
                     "Vague": s["share_vague"], "Quantified": s["share_quantified"], "Verified": s["share_verified"],
                     "Targets": s["targets"], "With baseline": s["targets_with_baseline"]})
    if sc is not None and not case_key:
        rows.append({"Report": f"{label} (yours)", "Sector": "–", "Grade": sc["report_grade"], "Score": sc["report_score"],
                     "Claims": sc["claims"], "Median specificity": sc["median_specificity"], "Vague": sc["share_vague"],
                     "Quantified": sc["share_quantified"], "Verified": sc["share_verified"], "Targets": sc["targets"],
                     "With baseline": sc["targets_with_baseline"]})
    cmp = pd.DataFrame(rows)
    st.dataframe(cmp.assign(**{c: cmp[c] * 100 for c in ("Vague", "Quantified", "Verified")}), hide_index=True,
                 width="stretch", column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Median specificity": st.column_config.NumberColumn(format="%.0f"),
        **{c: st.column_config.NumberColumn(c, format="%.0f%%") for c in ("Vague", "Quantified", "Verified")}})
    mixes = [r["mix"] for r in CASES["reports"].values()]
    if sc is not None and not case_key:
        mixes.append(claim_mix(df))
    mix_col = {"vague_only": ORANGE, "mixed": BLUE, "specific": AQUA, "neutral": "#b9b7b0"}
    names = [n.replace(" Plc", "").replace(" Holdings", "").replace(" Energy", "").replace(" Cement", "")
             if MOBILE else n for n in cmp.Report][::-1]
    min_lab = 0.12 if MOBILE else 0.06
    fig = go.Figure()
    for k, lab in MIX:
        vals = [m[k]["share"] for m in mixes][::-1]
        cnts = [m[k]["claims"] for m in mixes][::-1]
        fig.add_trace(go.Bar(y=names, x=vals, orientation="h", name=lab, marker_color=mix_col[k],
                             marker_line=dict(color="white", width=1.5), customdata=cnts,
                             text=[f"{v:.0%}" if v >= min_lab else "" for v in vals], textposition="inside", textangle=0,
                             insidetextanchor="middle", textfont=dict(color=INK if k == "neutral" else "white", size=12),
                             hovertemplate="%{y}<br>" + lab + ": %{x:.0%} (%{customdata} claims)<extra></extra>"))
    no_ev = [m["vague_only"]["share"] + m["neutral"]["share"] for m in mixes]
    style(fig, f"{min(no_ev):.0%}–{max(no_ev):.0%} of claims give no verifiable evidence",
          "Each claim counted once, so each bar = 100%. Evidence = number + unit, year, baseline, scope, verification "
          "or concrete action.", height=150 + 70 * len(mixes), legend=True)
    fig.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config=CFG)
    best_q = cmp.loc[cmp.Quantified.idxmax()]
    most_v = cmp.loc[cmp.Vague.idxmax()]
    st.markdown(f'''<div class="card"><h4>What stands out</h4>
      <div class="msg"><div class="n">1</div><div><div class="t">Most claims can't be checked</div><div class="b">
      {min(no_ev):.0%}–{max(no_ev):.0%} of environmental claims give no verifiable evidence (no number, year, baseline,
      scope, verification or concrete action). Only {cmp.Quantified.min():.0%}–{cmp.Quantified.max():.0%} carry a number
      with a unit ({html.escape(best_q.Report)} leads), and external verification is mentioned in under
      {cmp.Verified.max():.0%}.</div></div></div>
      <div class="msg"><div class="n">2</div><div><div class="t">Generic green language dominates where numbers are
      missing</div><div class="b">{html.escape(most_v.Report)} uses vague terms in {most_v.Vague:.0%} of its
      claims.</div></div></div>
      <div class="msg"><div class="n">3</div><div><div class="t">Targets lack baselines</div><div class="b">Of
      {int(cmp.Targets.sum())} measurable targets found, {int(cmp["With baseline"].sum())} state a baseline year, which
      IFRS S2 (adopted in Nigeria through the FRC roadmap) requires.</div></div></div>
      <div class="muted">Report years differ (2023–2025) and so do the report types (standalone vs annual-report
      section), so compare with care.</div></div>''', unsafe_allow_html=True)

# ======================================================================== METHOD
with t_meth:
    cd, gr = METR["claim_detector"], METR["greenwashing_risk"]
    section("How it works")
    a, b = st.columns(2, gap="large")
    a.markdown(f"""
**1. Read the report.** pypdf extracts text page by page, then rule-based splitting removes tables, headings and
infographic fragments.

**2. Find environmental claims.** TF-IDF (word + character n-grams) with logistic regression, trained on the
expert-annotated *environmental claims* dataset (Stammbach et al., ACL 2023; 2,647 sentences from annual reports,
sustainability reports and earnings calls). Test set: **macro-F1 {cd['macro_f1']:.3f}**, accuracy {cd['accuracy_micro_f1']:.3f},
ROC-AUC {cd['roc_auc']:.3f}, against the paper's TF-IDF SVM {cd['paper_test']['TF-IDF SVM']['macro_f1']:.3f},
ClimateBERT {cd['paper_test']['ClimateBERT']['macro_f1']:.3f} and RoBERTa-large {cd['paper_test']['RoBERTa-large']['macro_f1']:.3f}.

**3. Score specificity (0–100).** Each claim starts at 50. Thirteen transparent signals add or subtract points: vague
terms, absolute or neutrality claims, superlatives, overgeneralisation, exaggerated impact, offsets, aspirational
language, numbers, target year, baseline, scope, third-party verification and concrete action. Rules map to the EU
Empowering Consumers Directive (2024/825), the UK CMA Green Claims Code, the GHG Protocol and IFRS S2.

**4. Language-risk model.** Logistic regression on the 13 signals plus words, trained on {gr['n']} claims labelled
greenwashing / not greenwashing (EmeraldMind, 2026). 5-fold CV **ROC-AUC {gr['signals_plus_words_logreg_cv']['roc_auc']:.2f}**
(rules alone {gr['rule_score_only']['roc_auc']:.2f}).

**5. Targets.** Regex extraction of target year, baseline, percentage and scope, plus a completeness check.
""")
    b.markdown(f"""
**Limitations: read before using**
- **Not a verdict.** Real greenwashing assessment needs outside evidence (actual emissions, capex and news). A specific,
  quantified claim can still be false, and a vague one can be true.
- The language-risk model is trained on **short, semi-synthetic, news-style claims** about US/EU brands. Treat it as a
  secondary signal (AUC {gr['signals_plus_words_logreg_cv']['roc_auc']:.2f}).
- The claim detector misses some specific disclosures, so measurable targets and strongly evidenced numeric statements are
  added as claims to avoid penalising good reporting.
- **PDF extraction** struggles with scanned pages, multi-column layouts and infographics. Numbers in tables aren't analysed.
- English only. Signal weights are normative (policy-based), not fitted to data.

**Data**: environmental claims dataset © the authors (research use; commercial use requires permission) ·
EmeraldMind (Apache 2.0) · case-study reports are public PDFs, not redistributed; only short quoted excerpts with page
numbers are shown.

[Full methodology →]({REPO}/blob/master/docs/methodology.md)
""")
    with st.expander("Signal definitions, weights and references"):
        st.dataframe(pd.read_csv(ROOT / "data" / "reference" / "red_flags.csv").drop(columns=["pattern"]),
                     hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- audit note + footer
def audit_html() -> str:
    def q(rows):
        return "".join(f"<li>“{html.escape(r['sentence'])}” <i>(p. {r['page']}, {r['specificity']}/100)</i></li>" for r in rows)
    if case_key:
        weak_, strong_ = CASES["reports"][case_key]["weakest"][:5], CASES["reports"][case_key]["strongest"][:5]
    else:
        weak_, strong_ = weak[:5], strong[:5]
    trows = "" if targets is None or targets.empty else "".join(
        f"<tr><td>{int(r.page)}</td><td>{html.escape(str(r.type))}</td><td>{'' if r.target_year != r.target_year or r.target_year is None else int(r.target_year)}</td>"
        f"<td>{'' if r.baseline_year != r.baseline_year or r.baseline_year is None else int(r.baseline_year)}</td><td>{r.completeness:.0%}</td>"
        f"<td>{html.escape(r.sentence)}</td></tr>" for r in targets.itertuples())
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Claims audit: {html.escape(label)}</title><style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#0b0b0b;max-width:900px;margin:32px auto;padding:0 24px;line-height:1.5}}
h1{{color:{FOREST};margin-bottom:4px}} .sub{{color:#52514e}} .k{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:20px 0}}
.k div{{border:1px solid #e7e5e0;border-radius:10px;padding:10px}} .k b{{display:block;font-size:20px}} .k span{{font-size:11px;color:#52514e;text-transform:uppercase}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border-bottom:1px solid #e7e5e0;padding:6px;text-align:left;vertical-align:top}}
.note{{font-size:12px;color:#52514e;border-top:1px solid #e7e5e0;margin-top:24px;padding-top:10px}} li{{margin-bottom:6px}}</style></head><body>
<h1>Environmental claims audit</h1><div class="sub"><b>{html.escape(label)}</b> · generated {dt.date.today():%d %B %Y} · grade
<b>{sc['report_grade']}</b> ({GRADE_TXT[sc['report_grade']]}), score {sc['report_score']:.0f}/100</div>
<div class="k"><div><span>Claims</span><b>{sc['claims']}</b></div><div><span>Median specificity</span><b>{sc['median_specificity']:.0f}/100</b></div>
<div><span>Vague wording</span><b>{pct(sc['share_vague'])}</b></div><div><span>Quantified</span><b>{pct(sc['share_quantified'])}</b></div>
<div><span>Verified</span><b>{pct(sc['share_verified'])}</b></div></div>
<h3>Least specific claims</h3><ul>{q(weak_)}</ul><h3>Most specific claims</h3><ul>{q(strong_)}</ul>
<h3>Measurable targets</h3>{'<p>None found.</p>' if not trows else '<table><tr><th>Page</th><th>Type</th><th>Year</th><th>Baseline</th><th>Complete</th><th>Text</th></tr>' + trows + '</table>'}
<div class="note">{DISCLAIMER.replace('**', '')} Method: {REPO}. Prepared with the Greenwashing Detector by Daniel Oyewale, Petgrave.io.</div></body></html>"""


if sc is not None:
    with st.sidebar:
        st.download_button("⬇ Claims audit note (HTML, print to PDF)", audit_html().encode(), "claims_audit.html",
                           "text/html", width="stretch")

st.markdown(f"""<div class="foot"><b>Data:</b> Stammbach et al. (2023) environmental claims dataset · EmeraldMind
greenwashing claims (2026, Apache 2.0) · company reports: Access Holdings 2025, Dangote Cement 2024, Seplat Energy 2023
(public PDFs, excerpts only). <b>Rules:</b> EU Directive 2024/825 · UK CMA Green Claims Code · GHG Protocol · IFRS S2.<br>
{DISCLAIMER.replace('**', '')} · <a href="{REPO}">Code &amp; methodology</a> · Built by Daniel Oyewale, Petgrave.io</div>""",
            unsafe_allow_html=True)

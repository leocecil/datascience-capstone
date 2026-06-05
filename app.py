import streamlit as st
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Background ── */
.stApp {
    background:
        linear-gradient(rgba(8, 15, 30, 0.82), rgba(8, 15, 30, 0.88)),
        url("https://images.unsplash.com/photo-1566073771259-6a8506099945?q=80&w=1920&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Global text ── */
html, body, p, label {
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0;
}

h1 { font-family: 'Playfair Display', serif !important; }
h2, h3, h4 { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; color: #f8fafc !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
[aria-selected="true"] {
    background: rgba(56,189,248,0.18) !important;
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}
[data-baseweb="tab-panel"] { padding-top: 2rem !important; }

/* ── Metric cards ── */
[data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.8rem !important; letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { color: #38bdf8 !important; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem !important;
    backdrop-filter: blur(8px);
}

/* ── Info / warning / error / success boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    backdrop-filter: blur(6px) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] > details {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

/* ── Tables ── */
[data-testid="stTable"] table {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stTable"] th {
    background: rgba(56,189,248,0.15) !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
}
[data-testid="stTable"] td { color: #e2e8f0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.1) !important; margin: 2rem 0 !important; }

/* ── Custom section card ── */
.section-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
}

/* ── Numbered step badge ── */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    color: #0f172a;
    font-weight: 700;
    font-size: 0.75rem;
    border-radius: 50%;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    margin-right: 8px;
}

/* ── Tag pill ── */
.pill {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.35);
    color: #38bdf8;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px;
}

/* ── Model comparison rows ── */
.model-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
}
.model-winner { border-color: rgba(56,189,248,0.45); background: rgba(56,189,248,0.06); }
.model-name { font-weight: 600; min-width: 160px; font-size: 0.95rem; }
.model-badge-winner { background: #0ea5e9; color: #0f172a; border-radius: 6px; padding: 2px 9px; font-size: 0.72rem; font-weight: 700; }
.model-badge-alt    { background: rgba(255,255,255,0.12); color: #94a3b8; border-radius: 6px; padding: 2px 9px; font-size: 0.72rem; }
.stat { text-align: center; min-width: 80px; }
.stat-val { font-size: 1.1rem; font-weight: 700; color: #f8fafc; }
.stat-lbl { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Timeline ── */
.timeline-item {
    display: flex;
    gap: 1.2rem;
    margin-bottom: 1.4rem;
    align-items: flex-start;
}
.timeline-dot {
    flex-shrink: 0;
    width: 36px; height: 36px;
    background: linear-gradient(135deg,#0ea5e9,#38bdf8);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; color: #0f172a; font-size: 0.85rem;
}
.timeline-content h4 { margin: 0 0 4px; font-size: 0.95rem; color: #f8fafc !important; }
.timeline-content p  { margin: 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.5; }

/* ── Feature chip grid ── */
.chip-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.chip {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #cbd5e1;
}
.chip.num  { border-color: rgba(56,189,248,0.3);  background: rgba(56,189,248,0.07);  color: #7dd3fc; }
.chip.cat  { border-color: rgba(168,85,247,0.3);  background: rgba(168,85,247,0.07); color: #c4b5fd; }
</style>
""", unsafe_allow_html=True)


# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 3rem 0 1.5rem; text-align: center;">
  <p style="color:#38bdf8; font-size:0.85rem; letter-spacing:0.18em; text-transform:uppercase; margin-bottom:0.5rem;">
    Group 2 · Izin Tampil · Capstone Project
  </p>
  <h1 style="font-family:'Playfair Display',serif; font-size:clamp(2.2rem,5vw,3.6rem);
             font-weight:900; color:#f8fafc; line-height:1.15; margin:0 0 0.8rem;">
    Hotel Booking<br>
    <span style="color:#38bdf8;">Cancellation Predictor</span>
  </h1>
  <p style="color:#94a3b8; max-width:540px; margin:0 auto; font-size:1rem; line-height:1.6;">
    Machine learning–powered risk scoring to help hotels act before revenue is lost.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("hotel_bookings.csv")

df = load_data()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_home, tab_crisp, tab_predict = st.tabs(["🏠  Home", "📊  CRISP-DM", "🤖  Predict"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:

    # KPI row
    total   = len(df)
    cancel  = df["is_canceled"].sum()
    rate    = cancel / total * 100
    avg_adr = df["adr"].median()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Bookings",      f"{total:,}")
    k2.metric("Cancellations",        f"{cancel:,}")
    k3.metric("Cancellation Rate",    f"{rate:.1f}%")
    k4.metric("Median ADR",           f"${avg_adr:.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # What this app does
    st.markdown("""
    <div class="section-card">
      <h3 style="margin-top:0;">What This App Does</h3>
      <p style="color:#94a3b8; margin-bottom:1rem;">
        Every last-minute cancellation means an empty room and unrecoverable revenue.
        This tool uses a trained <strong style="color:#38bdf8;">Random Forest model</strong>
        to flag high-risk reservations <em>before</em> guests cancel — giving your team time to act.
      </p>
      <div style="display:flex; flex-wrap:wrap; gap:10px;">
        <div class="pill">📉 Revenue Protection</div>
        <div class="pill">📋 Smarter Overbooking</div>
        <div class="pill">🎯 Targeted Retention</div>
        <div class="pill">📅 Occupancy Forecasting</div>
        <div class="pill">🧹 Resource Planning</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset preview
    st.markdown("### 📋 Dataset Preview")
    st.caption("First 10 rows of the raw hotel bookings dataset (119,390 records · 32 columns)")
    st.dataframe(df.head(10), use_container_width=True, height=280)

    # Team
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card" style="text-align:center;">
      <p style="color:#94a3b8; font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;">The Team</p>
      <p style="font-size:1rem; color:#e2e8f0; margin:0;">
        Cecilia Agusta Leo &nbsp;·&nbsp; Jocelyn Jolie &nbsp;·&nbsp; Putu Diahloka Mahaputri
      </p>
      <p style="color:#475569; font-size:0.8rem; margin-top:6px;">
        Dataset: Hotel Booking Demand — Kaggle (Jesse Mostipak)
      </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CRISP-DM
# ══════════════════════════════════════════════════════════════════════════════
with tab_crisp:

    # CRISP-DM step nav
    steps = ["1 · Business Understanding", "2 · Data Understanding",
             "3 · Data Preparation", "4 · Modeling", "5 · Evaluation", "6 · Deployment"]
    chosen = st.radio("Jump to phase:", steps, horizontal=True, label_visibility="collapsed")

    st.markdown("---")

    # ── 1. BUSINESS UNDERSTANDING ─────────────────────────────────────────────
    if chosen == steps[0]:
        st.markdown("## 🏢 Business Understanding")
        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.error("""
**The Problem**

Last-minute booking cancellations leave rooms empty with no chance to rebook,
while sunk costs (staff, utilities, prep) have already been spent.
A high cancellation rate also makes staffing, provisioning, and scheduling
extremely hard to plan.
""")
            st.markdown("**Project Goal**")
            st.info("""
Predict whether a booking will be **canceled** before it happens,
using data available at reservation time — so staff can intervene early.
""")

        with col_r:
            st.markdown("**Business Value**")
            items = [
                ("📦", "Inventory Management",  "Calibrate overbooking levels precisely."),
                ("🎯", "Proactive Retention",    "Offer incentives to high-risk guests early."),
                ("💰", "Revenue Forecasting",    "Factor expected cancellations into nightly projections."),
                ("🧹", "Resource Allocation",    "Schedule cleaning & staff by booking confidence."),
            ]
            for icon, title, desc in items:
                st.markdown(f"""
                <div class="timeline-item">
                  <div class="timeline-dot">{icon}</div>
                  <div class="timeline-content">
                    <h4>{title}</h4><p>{desc}</p>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Success Metrics**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recall (target)",    "≥ 75%")
        m2.metric("Precision (target)", "≥ 62%")
        m3.metric("F1-Score (target)",  "≥ 0.68")
        m4.metric("Accuracy (target)",  "≥ 75%")

        st.caption("Recall is the primary driver — missing a real cancellation costs more than a false alarm.")

    # ── 2. DATA UNDERSTANDING ─────────────────────────────────────────────────
    elif chosen == steps[1]:
        st.markdown("## 🔍 Data Understanding")
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records",    f"{len(df):,}")
        c2.metric("Total Columns",    len(df.columns))
        c3.metric("Missing Values",   f"{df.isnull().sum().sum():,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Missing value breakdown
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing):
            st.markdown("**Missing Value Breakdown**")
            cols = st.columns(len(missing))
            for col, (name, cnt) in zip(cols, missing.items()):
                col.metric(name, f"{cnt:,}")

        st.markdown("---")

        # Charts
        st.markdown("**📊 Univariate Analysis**")
        st.image("univariate.png", use_container_width=True)
        st.caption("Key observations: cancellations are a minority class; ADR is right-skewed; lead time is widely distributed.")

        st.markdown("---")
        st.markdown("**📈 Multivariate Analysis**")
        st.image("multivariate.png", use_container_width=True)
        st.caption("Longer lead time → higher cancellation rate. Repeat guests cancel less. Deposit type strongly separates classes.")

        st.markdown("---")
        st.markdown("**🔥 Correlation Heatmap**")
        st.image("heatmap.png", use_container_width=True)
        st.caption("Most correlations stay below ±0.30 — low multicollinearity. Features were chosen on business relevance, not correlation alone.")

    # ── 3. DATA PREPARATION ───────────────────────────────────────────────────
    elif chosen == steps[2]:
        st.markdown("## 🛠 Data Preparation")
        st.markdown("<br>", unsafe_allow_html=True)

        # Feature chips
        st.markdown("**Selected Features**")
        st.markdown("""
        <div class="chip-grid">
          <span class="chip num">lead_time</span>
          <span class="chip num">total_of_special_requests</span>
          <span class="chip num">required_car_parking_spaces</span>
          <span class="chip num">booking_changes</span>
          <span class="chip num">previous_cancellations</span>
          <span class="chip num">has_agent</span>
          <span class="chip cat">deposit_type</span>
          <span class="chip cat">country</span>
          <span class="chip cat">market_segment</span>
          <span class="chip cat">distribution_channel</span>
          <span class="chip cat">hotel</span>
          <span class="chip cat">customer_type</span>
        </div>
        <p style="margin-top:10px; font-size:0.8rem; color:#64748b;">
          <span class="chip num" style="font-size:0.7rem;">blue</span> Numerical &nbsp;
          <span class="chip cat" style="font-size:0.7rem;">purple</span> Categorical
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Cleaning & Transformation Steps**")

        summary_df = pd.DataFrame({
            "Issue": [
                "Missing Values", "Duplicate Records", "Data Leakage",
                "Feature Engineering", "Invalid Records", "Outliers", "Categorical Features"
            ],
            "Action": [
                "Median / 'Unknown' imputation", "Removed", "Dropped leakage columns",
                "Created new features", "Removed zero-guest rows", "Threshold filtering",
                "One-Hot Encoding"
            ],
            "Why": [
                "Keep dataset complete", "Avoid repeated observations",
                "Prevent inflated accuracy", "Increase signal", "Remove impossible bookings",
                "Reduce noise & skew", "Convert text → numbers"
            ]
        })
        st.table(summary_df)

        st.markdown("---")
        st.markdown("**🎯 Target Variable**")
        col_a, col_b = st.columns(2)
        col_a.success("**0** — Booking completed (Not Canceled)")
        col_b.error("**1** — Booking canceled")
        st.caption("Binary classification target: `is_canceled`")

    # ── 4. MODELING ───────────────────────────────────────────────────────────
    elif chosen == steps[3]:
        st.markdown("## 🌲 Modeling")
        st.markdown("<br>", unsafe_allow_html=True)

        st.success("**Selected Model: Random Forest Classifier**  |  Train/Test split: 80:20  |  Random state: 42")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Why Random Forest?**")

        reasons = [
            ("🔀", "Non-Linear Patterns",      "Captures complex interactions between booking attributes."),
            ("📦", "Mixed Feature Types",       "Handles numerical + encoded categorical features natively."),
            ("🛡️", "Robust to Overfitting",     "Ensemble of trees generalizes better than a single tree."),
            ("📊", "Feature Importance",        "Explains which attributes drive cancellation risk most."),
            ("⚡", "No Distribution Assumption","Works without requiring Gaussian or other distributions."),
        ]
        for icon, title, desc in reasons:
            st.markdown(f"""
            <div class="timeline-item">
              <div class="timeline-dot">{icon}</div>
              <div class="timeline-content"><h4>{title}</h4><p>{desc}</p></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Why Not the Others?**")
        rej_col1, rej_col2 = st.columns(2)
        with rej_col1:
            with st.expander("❌ Logistic Regression"):
                st.write("Assumes mostly linear relationships. Misses complex feature interactions. Weakest Recall (55.87%) in our tests.")
            with st.expander("❌ K-Nearest Neighbors (KNN)"):
                st.write("Slow prediction on large datasets. Sensitive to scaling & noise. Poor interpretability for business use.")
        with rej_col2:
            with st.expander("❌ Naive Bayes"):
                st.write("Highest Recall (86.59%) but Precision only ~50% — half of cancellation flags are false alarms, eroding staff trust.")

    # ── 5. EVALUATION ─────────────────────────────────────────────────────────
    elif chosen == steps[4]:
        st.markdown("## 📊 Evaluation")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Model Comparison**")

        # ── Angka diupdate sesuai hasil ipynb ──
        models = [
            {"name": "Random Forest", "recall": "70.17%", "precision": "81.65%", "f1": "0.7548", "acc": "83.10%", "winner": True,  "note": "Best balance of precision & recall → selected for production"},
            {"name": "KNN",           "recall": "68.89%", "precision": "81.32%", "f1": "0.7459", "acc": "82.61%", "winner": False, "note": "High precision, but misses more cancellations than Random Forest"},
            {"name": "Naive Bayes",   "recall": "86.59%", "precision": "50.04%", "f1": "0.6343", "acc": "63.00%", "winner": False, "note": "Best recall, but ~50% false alarm rate"},
            {"name": "Logistic Reg.", "recall": "55.87%", "precision": "80.50%", "f1": "0.6596", "acc": "78.63%", "winner": False, "note": "Too conservative — misses nearly half of cancellations"},
        ]

        for m in models:
            cls = "model-row model-winner" if m["winner"] else "model-row"
            badge = '<span class="model-badge-winner">✓ SELECTED</span>' if m["winner"] else '<span class="model-badge-alt">compared</span>'
            st.markdown(f"""
            <div class="{cls}">
              <div class="model-name">{m["name"]} {badge}</div>
              <div class="stat"><div class="stat-val">{m["recall"]}</div><div class="stat-lbl">Recall</div></div>
              <div class="stat"><div class="stat-val">{m["precision"]}</div><div class="stat-lbl">Precision</div></div>
              <div class="stat"><div class="stat-val">{m["f1"]}</div><div class="stat-lbl">F1</div></div>
              <div class="stat"><div class="stat-val">{m["acc"]}</div><div class="stat-lbl">Accuracy</div></div>
              <div style="flex:1; color:#64748b; font-size:0.82rem; padding-left:0.5rem;">{m["note"]}</div>
            </div>""", unsafe_allow_html=True)

        st.image("model_comparison.png", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Final Model Performance — Random Forest**")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Recall",    "70.17%")
        e2.metric("Precision", "81.65%")
        e3.metric("F1-Score",  "0.7548")
        e4.metric("Accuracy",  "83.10%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""
**Why Random Forest wins:** While Naive Bayes catches more cancellations (86.59% recall),
its near-coin-flip precision (50.04%) means half its alerts are false alarms — eroding staff trust
and wasting retention resources. Random Forest provides a reliable balance hotels can act on confidently.

> 💡 If interventions later shift to low-cost automations (e.g., email-only), Naive Bayes becomes worth revisiting.
""")

    # ── 6. DEPLOYMENT ─────────────────────────────────────────────────────────
    elif chosen == steps[5]:
        st.markdown("## 🚀 Deployment")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**System Workflow**")
        steps_flow = [
            ("1", "User Input",       "Staff enters reservation details into the app form."),
            ("2", "Validation",       "Inputs are checked for validity (no negative ADR, guests ≥ 1, etc.)."),
            ("3", "Preprocessing",    "Raw inputs are encoded and scaled to match training format."),
            ("4", "Model Inference",  "Random Forest outputs a cancellation probability (0–100%)."),
            ("5", "Threshold Apply",  "Operator adjusts the decision threshold to trade Precision vs. Recall."),
            ("6", "Risk Display",     "The app shows final risk level + recommended action."),
        ]
        for num, title, desc in steps_flow:
            st.markdown(f"""
            <div class="timeline-item">
              <div class="timeline-dot">{num}</div>
              <div class="timeline-content"><h4>{title}</h4><p>{desc}</p></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Real-World Applications**")
            st.markdown("""
- Cancellation risk dashboard for front-desk staff  
- Overbooking strategy calibration  
- Automated retention email triggers  
- Shift & cleaning schedule optimization  
- Monthly revenue forecasting reports  
""")
        with col_b:
            st.markdown("**Adjustable Decision Threshold**")
            st.info("""
**Peak season** → lower threshold → catch more cancellations (higher Recall).  
**Normal season** → higher threshold → fewer false alarms (higher Precision).  

The threshold slider lets hotel management adapt the model's aggressiveness to current business priorities.
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT (placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
      <div style="font-size:3.5rem; margin-bottom:1rem;">🤖</div>
      <h2 style="font-family:'Playfair Display',serif; font-size:2rem; color:#f8fafc; margin-bottom:0.5rem;">
        Prediction Engine
      </h2>
      <div style="background:rgba(56,189,248,0.08); border:1px dashed rgba(56,189,248,0.35);
                  border-radius:16px; padding:2rem 2.5rem; display:inline-block; text-align:left; max-width:420px;">
        <p style="color:#94a3b8; font-size:0.85rem; margin:0 0 0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">Planned Features</p>
        <ul style="color:#cbd5e1; font-size:0.9rem; line-height:2; margin:0; padding-left:1.2rem;">
          <li>Booking details input form</li>
          <li>Real-time cancellation probability</li>
          <li>Adjustable risk threshold slider</li>
          <li>Risk level badge (Low / Medium / High)</li>
          <li>Top contributing features (SHAP)</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Teaser metrics
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#475569; font-size:0.85rem;'>Model performance when deployed</p>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Accuracy",  "83.10%")
    p2.metric("F1-Score",  "0.7548")
    p3.metric("Recall",    "70.17%")
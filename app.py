import streamlit as st
import pandas as pd
import pickle

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    with open('top_countries.pkl', 'rb') as f:
        top_countries = pickle.load(f)
    return model, model_columns, top_countries

model, model_columns, top_countries = load_model()

def predict(lead_time, total_special_requests, parking, booking_changes,
            hotel, prev_cancellations, has_agent,
            deposit_type, country, market_segment, distribution_channel, customer_type):
    country = country if country in top_countries else 'Other'
    raw = pd.DataFrame([{
        'lead_time': lead_time,
        'total_of_special_requests': total_special_requests,
        'required_car_parking_spaces': parking,
        'booking_changes': booking_changes,
        'hotel': 0 if hotel == 'Resort Hotel' else 1,
        'previous_cancellations': prev_cancellations,
        'has_agent': has_agent,
        'deposit_type': deposit_type,
        'country': country,
        'market_segment': market_segment,
        'distribution_channel': distribution_channel,
        'customer_type': customer_type,
    }])
    raw = pd.get_dummies(raw, columns=['market_segment', 'distribution_channel',
                                        'deposit_type', 'customer_type', 'country'],
                          drop_first=True)
    raw = raw.reindex(columns=model_columns, fill_value=0)
    return model.predict(raw)[0], model.predict_proba(raw)[0][1]

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
            
/* ── Predict tab ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    color: #94a3b8 !important; font-size: 0.82rem !important;
    font-weight: 500 !important; letter-spacing: 0.04em !important; text-transform: uppercase !important;
}
[data-testid="stNumberInput"] input {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
[data-baseweb="select"] div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {
    color: #0f172a !important;
    background: #f8fafc !important;
}
[data-baseweb="popover"] li:hover {
    background: #e0f2fe !important;
    color: #0ea5e9 !important;
}
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
    color: #0f172a !important; font-weight: 700 !important; font-size: 1rem !important;
    letter-spacing: 0.06em !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 0 !important;
}
div[data-testid="stButton"] > button:hover { box-shadow: 0 8px 25px rgba(56,189,248,0.35) !important; }
 
.form-section-label {
    color: #38bdf8; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    margin: 1.4rem 0 0.6rem;
    display: flex; align-items: center; gap: 8px;
}
.form-section-label::after { content: ''; flex: 1; height: 1px; background: rgba(56,189,248,0.2); }
 
.result-card { border-radius: 20px; padding: 2.2rem 2.4rem; margin-top: 1rem; }
.result-low  { background: rgba(16,185,129,0.12); border: 1.5px solid rgba(16,185,129,0.4); }
.result-med  { background: rgba(245,158,11,0.12);  border: 1.5px solid rgba(245,158,11,0.4); }
.result-high { background: rgba(239,68,68,0.12);   border: 1.5px solid rgba(239,68,68,0.4); }
 
.result-badge { display: inline-block; border-radius: 999px; padding: 5px 18px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem; }
.badge-low  { background: rgba(16,185,129,0.2);  color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
.badge-med  { background: rgba(245,158,11,0.2);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }
.badge-high { background: rgba(239,68,68,0.2);   color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
 
.prob-number { font-family: 'Playfair Display', serif; font-size: 4.5rem; font-weight: 900; line-height: 1; margin: 0.3rem 0; }
.prob-low  { color: #34d399; }
.prob-med  { color: #fbbf24; }
.prob-high { color: #f87171; }
 
.prob-bar-wrap { background: rgba(255,255,255,0.08); border-radius: 999px; height: 8px; margin: 1rem 0; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 999px; }
.fill-low  { background: linear-gradient(90deg,#059669,#34d399); }
.fill-med  { background: linear-gradient(90deg,#d97706,#fbbf24); }
.fill-high { background: linear-gradient(90deg,#dc2626,#f87171); }
 
.action-box { border-radius: 12px; padding: 1rem 1.3rem; margin-top: 1.2rem; font-size: 0.9rem; }
.action-low  { background: rgba(16,185,129,0.1);  border-left: 3px solid #34d399; color: #d1fae5 !important; }
.action-med  { background: rgba(245,158,11,0.1);  border-left: 3px solid #fbbf24; color: #fef3c7 !important; }
.action-high { background: rgba(239,68,68,0.1);   border-left: 3px solid #f87171; color: #fee2e2 !important; }

.action-low strong, .action-med strong, .action-high strong {
    color: #f8fafc !important;
}
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
# TAB 3 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem;">
      <p style="color:#38bdf8; font-size:0.8rem; letter-spacing:0.16em; text-transform:uppercase; margin-bottom:0.3rem;">
        Random Forest · 83.1% Accuracy
      </p>
      <h2 style="font-family:'Playfair Display',serif; font-size:2rem; color:#f8fafc; margin:0 0 0.4rem;">
        Cancellation Risk Engine
      </h2>
      <p style="color:#64748b; font-size:0.9rem; margin:0;">
        Fill in a booking's details below, or upload a CSV to score multiple bookings at once.
      </p>
    </div>
    """, unsafe_allow_html=True)
 
    mode = st.radio("Mode", ["✍️  Single Booking", "📂  Batch CSV Upload"],
                    horizontal=True, label_visibility="collapsed")
    st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
 
    # ════════════════════════════════════════════════════════════════════════
    # SINGLE BOOKING
    # ════════════════════════════════════════════════════════════════════════
    if mode == "✍️  Single Booking":
 
        form_col, result_col = st.columns([1.1, 0.9], gap="large")
 
        with form_col:
            st.markdown('<div class="form-section-label">🏨 Booking Details</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            hotel           = c1.selectbox("Hotel Type",   ["City Hotel", "Resort Hotel"])
            deposit_type    = c2.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
 
            c3, c4 = st.columns(2)
            market_segment  = c3.selectbox("Market Segment",
                                ["Online TA", "Offline TA/TO", "Direct", "Corporate",
                                 "Complementary", "Groups", "Aviation"])
            CHANNEL_MAP = {
                "Travel Agent / Tour Operator (TA/TO)": "TA/TO",
                "Direct": "Direct",
                "Corporate": "Corporate",
                "Global Distribution System (GDS)": "GDS",
                "Undefined": "Undefined"
            }
            channel_label   = c4.selectbox("Distribution Channel", list(CHANNEL_MAP.keys()))
            distribution_ch = CHANNEL_MAP[channel_label]
 
            c5, c6 = st.columns(2)
            customer_type   = c5.selectbox("Customer Type",
                                ["Transient", "Transient-Party", "Contract", "Group"])
            COUNTRY_MAP = {
                "Portugal": "PRT", "United Kingdom": "GBR", "France": "FRA",
                "Spain": "ESP", "Germany": "DEU", "Italy": "ITA", "Ireland": "IRL",
                "Belgium": "BEL", "Brazil": "BRA", "Netherlands": "NLD", "Other": "Other"
            }
            country_label = c6.selectbox("Country of Origin", list(COUNTRY_MAP.keys()))
            country_input = COUNTRY_MAP[country_label] 
            st.markdown('<div class="form-section-label">📅 Stay Info</div>', unsafe_allow_html=True)
            c7, c8 = st.columns(2)
            lead_time       = c7.number_input("Lead Time (days)", min_value=0, max_value=737, value=90,
                                              help="Days between booking date and arrival date")
            booking_changes = c8.number_input("Booking Changes", min_value=0, max_value=20, value=0,
                                              help="Number of amendments made to the booking")
 
            st.markdown('<div class="form-section-label">👤 Guest Profile</div>', unsafe_allow_html=True)
            c9, c10, c11 = st.columns(3)
            special_requests = c9.number_input("Special Requests",     min_value=0, max_value=5,  value=0)
            parking_spaces   = c10.number_input("Parking Spaces",      min_value=0, max_value=8,  value=0)
            prev_cancel      = c11.number_input("Prior Cancellations", min_value=0, max_value=26, value=0)
 
            has_agent_input = st.selectbox("Booked via Travel Agent?", ["Yes", "No"])
            has_agent_val   = 1 if has_agent_input == "Yes" else 0
 
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("⚙️  Adjust Decision Threshold", expanded=False):
                threshold = st.slider(
                    "Probability above this = flagged as CANCELED",
                    min_value=0.20, max_value=0.80, value=0.50, step=0.01, format="%.2f"
                )
                tc1, tc2 = st.columns(2)
                tc1.info(f"**Threshold: {threshold:.0%}**\nProbability ≥ {threshold:.0%} → Canceled")
                tc2.caption("**Peak season** → lower (0.35) → catch more risk\n\n**Normal season** → higher (0.60) → fewer false alarms")
 
            predict_btn = st.button("🔍  Analyse Booking Risk", use_container_width=True)
 
        with result_col:
            if predict_btn:
                pred_class, prob = predict(
                    lead_time, special_requests, parking_spaces, booking_changes,
                    hotel, prev_cancel, has_agent_val,
                    deposit_type, country_input, market_segment, distribution_ch, customer_type
                )
 
                if prob < 0.35:
                    tier, card_cls, badge_cls, bar_cls, fill_cls = "LOW", "result-low", "badge-low", "prob-low", "fill-low"
                    icon        = "✅"
                    action_cls  = "action-low"
                    action_text = "✅ <strong>Safe to proceed.</strong> Low cancellation probability — no intervention needed. Standard confirmation email recommended."
                    verdict     = "Not Likely to Cancel"
                elif prob < threshold:
                    tier, card_cls, badge_cls, bar_cls, fill_cls = "MEDIUM", "result-med", "badge-med", "prob-med", "fill-med"
                    icon        = "⚠️"
                    action_cls  = "action-med"
                    action_text = "⚠️ <strong>Monitor this booking.</strong> Consider a gentle re-confirmation message or a small loyalty incentive to reduce dropout risk."
                    verdict     = "Moderate Risk"
                else:
                    tier, card_cls, badge_cls, bar_cls, fill_cls = "HIGH", "result-high", "badge-high", "prob-high", "fill-high"
                    icon        = "🚨"
                    action_cls  = "action-high"
                    action_text = "🚨 <strong>High cancellation risk.</strong> Recommend immediate outreach — offer flexible upgrade, early check-in, or pre-arrival perks to retain this booking."
                    verdict     = "Likely to Cancel"
 
                pct = int(prob * 100)
 
                st.markdown(f"""
                <div class="result-card {card_cls}">
                  <span class="result-badge {badge_cls}">{icon} {tier} RISK</span>
                  <div style="color:#94a3b8; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.08em;">Cancellation Probability</div>
                  <div class="prob-number {bar_cls}">{pct}%</div>
                  <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.5rem;">{verdict}</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar-fill {fill_cls}" style="width:{pct}%;"></div>
                  </div>
                  <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#475569; margin-bottom:0.8rem;">
                    <span>0%</span><span>Threshold: {int(threshold*100)}%</span><span>100%</span>
                  </div>
                  <div class="action-box {action_cls}">{action_text}</div>
                </div>
                """, unsafe_allow_html=True)
 
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📌 Key Factors in This Prediction**")
 
                factors = []
                if lead_time > 150:
                    factors.append(("📅 Long Lead Time", f"{lead_time} days — bookings this far ahead cancel more often", "high"))
                elif lead_time < 30:
                    factors.append(("📅 Short Lead Time", f"{lead_time} days — last-minute bookings are usually more committed", "low"))
                if deposit_type == "Non Refund":
                    factors.append(("💳 Non-Refundable Deposit", "Paradoxically the strongest predictor of cancellation in this dataset", "high"))
                elif deposit_type == "No Deposit":
                    factors.append(("💳 No Deposit", "Zero financial commitment — slightly higher dropout risk", "med"))
                if prev_cancel > 0:
                    factors.append(("🔁 Previous Cancellations", f"Guest has canceled {prev_cancel}× before — strong risk signal", "high"))
                if special_requests > 1:
                    factors.append(("⭐ Special Requests", f"{special_requests} requests — guests with requests are more invested", "low"))
                if has_agent_val == 0:
                    factors.append(("🤝 No Agent", "Direct bookings without agents show slightly higher cancel rates", "med"))
                if not factors:
                    factors.append(("📊 Mixed Signals", "No single dominant factor — model used all features collectively", "med"))
 
                tier_color = {"high": "#f87171", "med": "#fbbf24", "low": "#34d399"}
                for fname, fdesc, ftier in factors:
                    st.markdown(f"""
                    <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:0.6rem;
                                background:rgba(255,255,255,0.04); border-radius:10px; padding:0.75rem 1rem;
                                border-left:3px solid {tier_color[ftier]};">
                      <div>
                        <div style="font-size:0.88rem; font-weight:600; color:#f1f5f9;">{fname}</div>
                        <div style="font-size:0.8rem; color:#94a3b8; margin-top:2px;">{fdesc}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
 
            else:
                st.markdown("""
                <div style="text-align:center; padding:4rem 2rem; border:1px dashed rgba(255,255,255,0.1);
                            border-radius:20px; background:rgba(255,255,255,0.02);">
                  <div style="font-size:3rem; margin-bottom:1rem;">🔍</div>
                  <p style="color:#475569; font-size:0.95rem; margin:0;">
                    Fill in the booking details on the left<br>and click <strong style="color:#38bdf8;">Analyse Booking Risk</strong>
                  </p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px; padding:1.2rem 1.4rem;">
                  <p style="color:#64748b; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin:0 0 0.8rem;">Model at a glance</p>
                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
                    <div><div style="color:#38bdf8;font-size:1.4rem;font-weight:700;">83.1%</div><div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;">Accuracy</div></div>
                    <div><div style="color:#38bdf8;font-size:1.4rem;font-weight:700;">0.755</div><div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;">F1-Score</div></div>
                    <div><div style="color:#38bdf8;font-size:1.4rem;font-weight:700;">70.2%</div><div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;">Recall</div></div>
                    <div><div style="color:#38bdf8;font-size:1.4rem;font-weight:700;">81.7%</div><div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;">Precision</div></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
 
    # ════════════════════════════════════════════════════════════════════════
    # BATCH CSV MODE
    # ════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("""
        <div class="section-card">
          <h3 style="margin-top:0;">📂 Batch Booking Scorer</h3>
          <p style="color:#94a3b8; margin-bottom:1rem;">
            Upload a CSV with multiple bookings and get a cancellation risk score for each row instantly.
            The CSV must contain these columns:
          </p>
          <div class="chip-grid">
            <span class="chip num">lead_time</span>
            <span class="chip num">total_of_special_requests</span>
            <span class="chip num">required_car_parking_spaces</span>
            <span class="chip num">booking_changes</span>
            <span class="chip num">previous_cancellations</span>
            <span class="chip num">has_agent</span>
            <span class="chip cat">hotel</span>
            <span class="chip cat">deposit_type</span>
            <span class="chip cat">country</span>
            <span class="chip cat">market_segment</span>
            <span class="chip cat">distribution_channel</span>
            <span class="chip cat">customer_type</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
        batch_threshold = st.slider(
            "Risk threshold for batch scoring",
            min_value=0.20, max_value=0.80, value=0.50, step=0.01, format="%.2f"
        )
 
        template_cols = ["lead_time","total_of_special_requests","required_car_parking_spaces",
                         "booking_changes","previous_cancellations","has_agent",
                         "hotel","deposit_type","country","market_segment",
                         "distribution_channel","customer_type"]
        template_df = pd.DataFrame({
            "lead_time": [90, 14],
            "total_of_special_requests": [0, 2],
            "required_car_parking_spaces": [0, 1],
            "booking_changes": [0, 1],
            "previous_cancellations": [0, 0],
            "has_agent": [1, 0],
            "hotel": ["City Hotel", "Resort Hotel"],
            "deposit_type": ["No Deposit", "Non Refund"],
            "country": ["PRT", "GBR"],
            "market_segment": ["Online TA", "Direct"],
            "distribution_channel": ["TA/TO", "Direct"],
            "customer_type": ["Transient", "Transient-Party"],
        })
 
        dl_col, up_col = st.columns([1, 2])
        with dl_col:
            st.download_button(
                "⬇️  Download CSV Template",
                data=template_df.to_csv(index=False),
                file_name="booking_template.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with up_col:
            uploaded = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")
 
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                missing_cols = [c for c in template_cols if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    with st.spinner(f"Scoring {len(batch_df):,} bookings…"):
                        probs = []
                        for _, row in batch_df.iterrows():
                            _, p = predict(
                                row["lead_time"], row["total_of_special_requests"],
                                row["required_car_parking_spaces"], row["booking_changes"],
                                row["hotel"], row["previous_cancellations"], int(row["has_agent"]),
                                row["deposit_type"], row["country"], row["market_segment"],
                                row["distribution_channel"], row["customer_type"],
                            )
                            probs.append(p)
 
                    batch_df["cancel_probability"]     = probs
                    batch_df["cancel_probability_pct"] = (batch_df["cancel_probability"] * 100).round(1).astype(str) + "%"
                    batch_df["risk_tier"] = batch_df["cancel_probability"].apply(
                        lambda p: "🔴 HIGH" if p >= batch_threshold else ("🟡 MEDIUM" if p >= 0.35 else "🟢 LOW")
                    )
                    batch_df["verdict"] = batch_df["cancel_probability"].apply(
                        lambda p: "Likely to Cancel" if p >= batch_threshold else ("Monitor" if p >= 0.35 else "Safe")
                    )
 
                    n_high = (batch_df["cancel_probability"] >= batch_threshold).sum()
                    n_med  = ((batch_df["cancel_probability"] >= 0.35) & (batch_df["cancel_probability"] < batch_threshold)).sum()
                    n_low  = (batch_df["cancel_probability"] < 0.35).sum()
                    avg_p  = batch_df["cancel_probability"].mean()
 
                    st.markdown("<br>", unsafe_allow_html=True)
                    bk1, bk2, bk3, bk4 = st.columns(4)
                    bk1.metric("Total Scored",  f"{len(batch_df):,}")
                    bk2.metric("🔴 High Risk",   f"{n_high:,}")
                    bk3.metric("🟡 Medium Risk", f"{n_med:,}")
                    bk4.metric("🟢 Low Risk",    f"{n_low:,}")
 
                    st.markdown(f"""
                    <div style="background:rgba(56,189,248,0.07); border:1px solid rgba(56,189,248,0.2);
                                border-radius:12px; padding:0.9rem 1.2rem; margin:0.8rem 0 1.2rem;">
                      <span style="color:#94a3b8; font-size:0.82rem;">Average cancellation probability: </span>
                      <strong style="color:#38bdf8; font-size:1rem;">{avg_p:.1%}</strong>
                    </div>
                    """, unsafe_allow_html=True)
 
                    display_df = batch_df[["risk_tier","cancel_probability_pct","verdict"] + template_cols].copy()
                    display_df.columns = ["Risk","Probability","Verdict"] + template_cols
                    st.dataframe(display_df, use_container_width=True, height=420)
 
                    st.download_button(
                        "⬇️  Download Scored Results",
                        data=batch_df.to_csv(index=False),
                        file_name="bookings_scored.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
 
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
 

    # Teaser metrics
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#475569; font-size:0.85rem;'>Model performance when deployed</p>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Accuracy",  "83.10%")
    p2.metric("F1-Score",  "0.7548")
    p3.metric("Recall",    "70.17%")
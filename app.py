import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Hotel Booking Analysis", layout="wide")

# ---- HEADER ----
st.title("🏨 Hotel Booking Predictions")
st.markdown("A machine learning-powered application to predict reservation cancellations based on historical data.")

# ---- HORIZONTAL NAVIGATION (Tabs) ----
tab_home, tab_crisp, tab_predict = st.tabs(["🏠 Home", "📊 CRISP-DM", "🤖 Predict"])

with tab_home:
    st.subheader("Welcome!")
    st.write("Use this application to analyze and mitigate hotel cancellation risks.")

    st.markdown("### Dataset Preview")
    df = pd.read_csv("hotel_bookings.csv")
    st.dataframe(df.head(10))

with tab_crisp:
    st.title("🔄 CRISP-DM Process")
    st.caption("Model development documentation based on the CRISP-DM standard framework.")
    
    # ---------------------------------------------------------
    # 1. BUSINESS UNDERSTANDING
    # ---------------------------------------------------------
    st.markdown("### 1️⃣ Business Understanding")
        
    st.markdown("#### 🚨 Problem Statement & Impact")

    st.error("""
    **Business Problem:** A hotel reservation that is canceled at the last minute represents a direct loss of revenue. 
    High rates of sudden booking cancellations disrupt revenue management and optimal room allocation.
    """)
    
    impact_col1, impact_col2 = st.columns(2)
    with impact_col1:
        st.markdown("""
        **Financial Impact:**
        * **Lost Income:** The room sits empty for the night, resulting in lost income that cannot be recovered.
        * **Sunk Costs:** The hotel has already allocated resources (cleaning staff, minimum room inventory, utilities) for that date.
        """)
    with impact_col2:
        st.markdown("""
        **Operational Impact:**
        * **Inconsistent Planning:** A high cancellation rate makes it difficult to plan staffing, breakfast provisioning, maintenance schedules, and event space bookings.
        """)
        
    st.markdown("---")
    st.markdown("#### 🎯 Project Objective")
    
    st.info("""
    The **primary business objective** of this project is to **predict whether a booking will be canceled** based on historical booking characteristics and customer behavior data.
    """)
    
    with st.expander("🔍 Why This Matters (Business Value)", expanded=True):
        st.markdown("""
        * **Optimized Inventory Management:** Manage overbooking levels more precisely, minimizing room losses and guest displacement.
        * **Proactive Mitigation:** Identify high-risk bookings early and implement targeted retention actions *(e.g., personalized offers or pre-payment confirmation)*.
        * **Financial Predictability:** Improve revenue forecasting accuracy by accounting for the expected number of cancellations each night.
        * **Resource Allocation:** Prioritize cleaning and room-assignment scheduling based on booking confidence.
        """)

    st.markdown("---")
    st.markdown("#### 🚀 Project Roadmap & Success Metrics")
    sub_tab_goals, sub_tab_criteria = st.tabs(["🎯 Project Goals (What We Achieve)", "🏆 Success Criteria (How We Measure)"])

    with sub_tab_goals:
        st.markdown("We framework our milestone into **5 Core Goals**, moving from model creation to flexible business decision support:")
        
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("""
            * **G1: Predictive Intelligence**
              Train a machine learning model that classifies a booking as *canceled* (`1`) or *not canceled* (`0`) using features available at booking time.
            * **G2: High Performance Benchmarks**
              Target solid predictive power on the minority (canceled) class on a held-out test set to ensure the AI doesn't just guess blindly.
            * **G3: Actionable Feature Insights**
              Identify the top 5–10 booking attributes most strongly associated with cancellation to guide operational policy changes.
            """)
        with g_col2:
            st.markdown("""
            * **G4: Interactive Application Delivery**
              Integrate the best-performing model into this Streamlit app so users can input booking details and receive a risk probability in real time.
            """)
            # Menyoroti Goal 5 kesukaanmu dengan visual Box Sukses/Hijau agar stand-out!
            st.success("""
            **🎯 G5: Threshold-Based Decision Support (Our Core Focus)**
            
            Allow the hotel operator to **adjust the classification decision threshold** in real-time. This lets management trade off *Precision* (few false alarms to avoid disturbing loyal guests) against *Recall* (catch as many cancellations as possible during high-peak seasons).
            """)

        with sub_tab_criteria:
            c_col1, c_col2 = st.columns([2, 3]) # Pembagian rasio kolom (kiri agak lebar)
        
        with c_col1:
            st.markdown("**📊 Quantitative Benchmark (Test Set Data Evaluation):**")
            # Menampilkan target metrik kuantitatif menggunakan kolom mini yang estetik
            m_row1_1, m_row1_2 = st.columns(2)
            m_row1_1.metric("F1-Score (Canceled)", "≥ 0.68")
            m_row1_2.metric("Precision", "≥ 0.62")
            
            m_row2_1, m_row2_2 = st.columns(2)
            m_row2_1.metric("Recall (Catch Rate)", "≥ 0.75")
            m_row2_2.metric("Overall Accuracy", "≥ 0.75")
            
        with c_col2:
            st.markdown("**🧠 Qualitative Standard (User Experience & Trust):**")
            st.markdown("""
            * **Explainable AI:** The model must be transparent. Feature importance or SHAP/LIME outputs must explain *why* a booking is flagged as high-risk.
            * **Sub-5s Latency:** The application must load and serve predictions within **5 seconds** on standard hardware.
            * **Bulletproof Input Validation:** The UI strictly prevents nonsensical inputs *(e.g., negative average daily rate (ADR) or adults ≤ 0)*.
            """)
    
    st.markdown("---") # Section divider untuk lanjut ke 2. Data Understanding

    # ---------------------------------------------------------
    # 2. DATA UNDERSTANDING
    # ---------------------------------------------------------
    st.markdown("### 2️⃣ Data Understanding")
    m1, m2 = st.columns(2)
    m1.metric("Total Data Columns", "32 Columns")
    m2.metric("Total Data Rows", "119,390 Rows")
    
    st.write("**Target Variable:** `is_canceled` (0 = Check-in, 1 = Canceled)")
    
    st.markdown("---")

    # ---------------------------------------------------------
    # 3. DATA PREPARATION
    # ---------------------------------------------------------
    st.markdown("### 3️⃣ Data Preparation")
    st.write("Data cleaning and preprocessing steps performed:")
    st.markdown("""
    * **Handling Missing Values:** Imputed missing values in `agent` and `company` columns with 0.
    * **Feature Encoding:** Transformed categorical data types using *One-Hot Encoding*.
    """)
    
    st.markdown("---")

    # ---------------------------------------------------------
    # 4. MODELING
    # ---------------------------------------------------------
    st.markdown("### 4️⃣ Modeling")
    st.write("The algorithm used is the **Random Forest Classifier** due to its robustness in handling non-linear relationships within customer behavior data.")
    
    # Remaining phases: 5 (Evaluation) and 6 (Deployment) can be appended below...

with tab_predict:
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    # Note: You can replace these placeholders with classification metrics later (e.g., Accuracy, F1-Score)
    m1.metric("Accuracy", "89.2%")
    m2.metric("F1-Score", "0.915")
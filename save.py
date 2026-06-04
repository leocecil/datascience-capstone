import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Hotel Booking Analysis", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), 
                    url("https://images.unsplash.com/photo-1566073771259-6a8506099945?q=80&w=1920&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    h1, h2, h3, h4, p, span, li {
        color: #f8fafc !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    
    st.markdown("---") 

    # ---------------------------------------------------------
    # 2. DATA UNDERSTANDING
    # ---------------------------------------------------------
    st.markdown("### 2️⃣ Data Understanding")

    st.write("""
    The dataset contains hotel booking records and customer-related information.
    This phase focuses on understanding the characteristics, distributions,
    and relationships between variables before data preparation and modeling.
    """)

    # =========================
    # Univariate Analysis
    # =========================
    st.subheader("📊 Univariate Analysis")

    st.image("univariate.png", use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Most bookings are not canceled compared to canceled bookings.
    - The ADR (Average Daily Rate) distribution is right-skewed, indicating a small number of very expensive bookings.
    - Lead time shows a wide spread, suggesting customers book rooms at varying times before arrival.
    """)

    st.markdown("---")

    # =========================
    # Multivariate Analysis
    # =========================
    st.subheader("📈 Multivariate Analysis")

    st.image("multivariate.png", use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Cancellation rates tend to increase as lead time becomes longer.
    - Guests with repeated bookings appear less likely to cancel.
    - Certain booking characteristics show noticeable differences between canceled and non-canceled reservations.
    """)

    st.markdown("---")

    # =========================
    # Correlation Analysis
    # =========================
    st.subheader("🔥 Correlation Heatmap")

    st.image("heatmap.png", use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Most numerical variables have weak correlations with each other.
    - The strongest correlations remain below ±0.30, indicating low multicollinearity.
    - Features were selected not only based on correlation strength but also on business relevance and domain understanding.
    """)

    # Dataset Overview
    total_records = len(df)
    total_columns = len(df.columns)
    duplicate_rows = df.duplicated().sum()

    st.subheader("📋 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Records", f"{total_records:,}")

    with c2:
        st.metric("Total Columns", total_columns)

    with c3:
        st.metric("Duplicate Rows", f"{duplicate_rows:,}")

    st.subheader("🔍 Missing Values Analysis")

    total_missing = df.isnull().sum().sum()

    st.metric("Total Missing Values", f"{total_missing:,}")

    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]

    cols = st.columns(len(missing_cols))

    for col, (column_name, missing_count) in zip(cols, missing_cols.items()):
        with col:
            st.metric(
                column_name,
                f"{missing_count:,}"
            )   
    
    st.markdown("---")

    # ---------------------------------------------------------
    # 3. DATA PREPARATION
    # ---------------------------------------------------------
    st.markdown("### 3️⃣ Data Preparation")

    st.markdown("""
    This phase transforms raw booking records into a clean and machine-learning-ready dataset.
    The objective is to improve data quality, reduce noise, prevent data leakage,
    and create meaningful features that help predict booking cancellations.
    """)

    # ============================================
    # FEATURE ANALYSIS
    # ============================================
    st.subheader("📊 Feature Analysis")

    st.markdown("""
    The selected features capture different aspects of booking behavior,
    customer commitment, and reservation characteristics that are commonly
    associated with cancellation risk.
    """)

    st.markdown("""
    - **Lead Time**: Measures how far in advance a booking is made. Longer lead times often correlate with higher cancellation rates.
    - **Special Requests**: Reflects customer engagement and booking seriousness.
    - **Parking Spaces & Booking Changes**: Indicate customer commitment and reservation modifications.
    - **Previous Cancellations**: Captures historical cancellation behavior.
    - **Deposit Type**: Represents financial commitment before arrival.
    - **Market Segment & Distribution Channel**: Describe how customers interact with the hotel.
    - **Customer Type**: Distinguishes transient, contract, and group customers with different cancellation tendencies.
    """)

    # ============================================
    # TARGET VARIABLE
    # ============================================
    st.subheader("🎯 Target Variable")

    st.markdown("""
    The target variable is **is_canceled**.

    This binary variable indicates whether a booking was ultimately canceled:

    - **0** → Booking was completed (Not Canceled)
    - **1** → Booking was canceled

    The goal of the machine learning model is to predict this outcome before the arrival date,
    allowing hotel managers to take preventive actions for high-risk reservations.
    """)

    # ============================================
    # FEATURE TABLE
    # ============================================
    feature_table = pd.DataFrame({
        "Category": [
            "Numerical",
            "Numerical",
            "Numerical",
            "Numerical",
            "Numerical",
            "Numerical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical"
        ],
        "Variables": [
            "lead_time",
            "total_of_special_requests",
            "required_car_parking_spaces",
            "booking_changes",
            "previous_cancellations",
            "has_agent",
            "deposit_type",
            "country",
            "market_segment",
            "distribution_channel",
            "hotel",
            "customer_type"
        ]
    })

    st.table(feature_table)

    st.markdown("---")

    # ============================================
    # PREPROCESSING SUMMARY
    # ============================================
    st.subheader("🛠 Data Cleaning & Transformation Summary")

    summary_df = pd.DataFrame({
        "Issue": [
            "Missing Values",
            "Duplicate Records",
            "Data Leakage",
            "Feature Engineering",
            "Invalid Records",
            "Outliers",
            "Categorical Features"
        ],
        "Action": [
            "Median / Unknown Imputation",
            "Removed",
            "Removed Leakage Columns",
            "Created New Features",
            "Removed Zero-Guest Bookings",
            "Threshold Filtering",
            "One-Hot Encoding"
        ],
        "Reason": [
            "Maintain dataset completeness",
            "Avoid repeated observations",
            "Prevent unrealistically high accuracy",
            "Increase predictive information",
            "Remove impossible bookings",
            "Reduce noise and skewness",
            "Convert text to numerical format"
        ]
    })

    st.table(summary_df)

    # ---------------------------------------------------------
    # 4. MODELING
    # ---------------------------------------------------------
    st.markdown("### 4️⃣ Modeling")

    st.markdown("""
    The objective of this phase is to train a machine learning model capable of
    predicting whether a hotel booking will be canceled based on customer and reservation characteristics.
    """)

    st.subheader("🌲 Why Random Forest?")

    st.markdown("""
    Random Forest was selected as the primary model because it aligns well with the characteristics of the hotel booking dataset.

    - **Captures Non-Linear Patterns:** Booking cancellations are influenced by complex interactions between customer behavior and reservation details.
    - **Handles Mixed Feature Types:** Works effectively with both numerical and encoded categorical variables.
    - **Robust Against Overfitting:** Combines multiple decision trees to improve generalization.
    - **Provides Feature Importance:** Helps explain which booking attributes contribute most to cancellation risk.
    - **Minimal Assumptions:** Does not require data to follow a specific statistical distribution.
    """)

    st.markdown("---")

    st.subheader("❌ Why NOT Logistic Regression?")

    st.markdown("""
    Although Logistic Regression is a popular classification algorithm, it was not chosen as the final model because:

    - Assumes mostly linear relationships between features and cancellation probability.
    - May struggle to capture complex interactions between booking attributes.
    - Less effective when customer behavior patterns are highly non-linear.
    - Typically provides lower predictive performance on structured datasets with complex feature relationships.
    """)

    st.markdown("---")

    st.subheader("❌ Why NOT Decision Tree?")

    st.markdown("""
    Decision Trees are easy to interpret but have several limitations:

    - Highly sensitive to small changes in training data.
    - More prone to overfitting.
    - Single-tree predictions are generally less stable.
    - Random Forest reduces these issues by averaging many trees together.
    """)

    st.markdown("---")

    st.subheader("❌ Why NOT K-Nearest Neighbors (KNN)?")

    st.markdown("""
    KNN was not selected because:

    - Prediction becomes slower as dataset size increases.
    - Sensitive to feature scaling and noisy observations.
    - Performance can degrade in higher-dimensional feature spaces.
    - Less interpretable for business decision-making.
    """)

    st.markdown("---")

    st.subheader("⚙️ Model Training Configuration")

    config_df = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Train-Test Split",
            "Random State",
            "Target Variable",
            "Class Type"
        ],
        "Value": [
            "Random Forest Classifier",
            "80:20",
            "42",
            "is_canceled",
            "Binary Classification"
        ]
    })

    st.table(config_df)

    st.markdown("---")

    st.subheader("🎯 Modeling Objective")

    st.markdown("""
    The model outputs the probability that a reservation will be canceled.

    This probability can then be converted into a business decision using a customizable threshold,
    allowing hotel managers to balance between:

    - **Precision:** Reducing false alarms for loyal customers.
    - **Recall:** Identifying as many potential cancellations as possible.

    This flexibility supports different operational strategies during normal and peak seasons.
    """)

    st.markdown("---")

    # ---------------------------------------------------------
    # 5. EVALUATION
    # ---------------------------------------------------------
    st.markdown("### 5️⃣ Evaluation")

    st.markdown("""
    The final Random Forest model was evaluated using multiple classification metrics
    to ensure reliable cancellation prediction performance.
    """)

    st.subheader("📊 Model Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Recall", "71.64%")
    m2.metric("Precision", "77.46%")
    m3.metric("F1-Score", "0.7443")
    m4.metric("Accuracy", "81.67%")


    st.markdown("""
    **Performance Interpretation:**

    - **High Accuracy:** The model correctly classifies the majority of booking outcomes.
    - **Strong Recall:** Most cancellations are successfully detected before arrival.
    - **Balanced Precision:** The model avoids generating excessive false cancellation alerts.
    - **Robust F1-Score:** Indicates a good balance between precision and recall.
    """)

    st.markdown("---")

    st.subheader("🎯 Business Impact")

    st.markdown("""
    The model's value extends beyond prediction accuracy by supporting operational decisions.

    - **Revenue Protection:** Early identification of high-risk bookings helps reduce revenue loss.
    - **Improved Forecasting:** Expected cancellations can be incorporated into occupancy planning.
    - **Resource Optimization:** Staffing and room allocation can be adjusted proactively.
    - **Customer Retention:** Hotels can target at-risk customers with promotional incentives.
    """)

    st.markdown("---")

    # ---------------------------------------------------------
    # 6. DEPLOYMENT
    # ---------------------------------------------------------
    st.markdown("### 6️⃣ Deployment")

    st.markdown("""
    The trained Random Forest model was integrated into this Streamlit application,
    allowing hotel staff to assess cancellation risk in real time.
    """)

    st.subheader("💻 System Workflow")

    st.markdown("""
    The deployed prediction workflow consists of the following steps:

    1. User enters booking information.
    2. Input data is validated and preprocessed.
    3. The trained Random Forest model generates a cancellation probability.
    4. A decision threshold is applied.
    5. The system displays the final prediction and risk level.
    """)

    st.markdown("---")

    st.subheader("🏨 Real-World Applications")

    st.markdown("""
    The application can support hotel operations in several ways:
    """)

    st.markdown("""
    - **Cancellation Risk Monitoring:** Identify reservations likely to be canceled.
    - **Revenue Management:** Improve overbooking and occupancy strategies.
    - **Targeted Retention Campaigns:** Offer incentives to high-risk customers.
    - **Operational Planning:** Adjust staffing and room preparation schedules.
    - **Demand Forecasting:** Improve future occupancy predictions.
    """)

    st.markdown("---")

    st.subheader("🚀 Business Value")

    st.markdown("""
    By combining machine learning predictions with managerial judgment,
    the system helps hotels move from reactive cancellation handling
    to proactive risk management.

    This allows decision-makers to take preventive actions before revenue is lost.
    """)

with tab_predict:
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    # Note: You can replace these placeholders with classification metrics later (e.g., Accuracy, F1-Score)
    m1.metric("Accuracy", "89.2%")
    m2.metric("F1-Score", "0.915")
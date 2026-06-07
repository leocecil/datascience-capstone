"""
generate_model.py
Run this script once to produce model.pkl for the Streamlit app.
It replicates the exact preprocessing and training pipeline from the notebook.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading dataset...")
url = "https://raw.githubusercontent.com/leocecil/datascience-capstone/main/hotel_bookings.csv"
df = pd.read_csv(url)

# ── 2. CLEANING ───────────────────────────────────────────────────────────────
print("Cleaning data...")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Convert date column
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# Fill missing values
df['children'] = df['children'].fillna(df['children'].median())
df['country']  = df['country'].fillna('Unknown')

# Drop data-leakage & personal columns
df.drop(
    columns=['reservation_status', 'reservation_status_date',
             'name', 'email', 'phone-number', 'credit_card'],
    inplace=True,
    errors='ignore'
)

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
# Binary: has_agent / has_company
df['has_agent']   = df['agent'].notnull().astype(int)
df['has_company'] = df['company'].notnull().astype(int)
df.drop(['agent', 'company'], axis=1, inplace=True)

# Total guests
df['total_guests'] = df['adults'] + df['children'].astype(int) + df['babies']

# Remove impossible bookings (zero guests)
df = df[(df['adults'] > 0) | (df['children'] > 0) | (df['babies'] > 0)]

# ── 4. FEATURE SELECTION ─────────────────────────────────────────────────────
column_kept = [
    'is_canceled',
    # Numerical
    'lead_time', 'total_of_special_requests', 'required_car_parking_spaces',
    'booking_changes', 'previous_cancellations', 'has_agent',
    # Categorical
    'deposit_type', 'country', 'market_segment', 'distribution_channel',
    'hotel', 'customer_type'
]
df = df[column_kept].copy()

# ── 5. ENCODING ───────────────────────────────────────────────────────────────
# Hotel: Resort Hotel → 0, City Hotel → 1
df['hotel'] = df['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})

# Country: keep top-10, rest → 'Other'
top_countries = df['country'].value_counts().nlargest(10).index
df['country'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')

# One-hot encode remaining categoricals
categorical_cols = ['market_segment', 'distribution_channel',
                    'deposit_type', 'customer_type', 'country']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ── 6. DEFINE X / y ───────────────────────────────────────────────────────────
numerical_columns = [
    'lead_time', 'total_of_special_requests', 'required_car_parking_spaces',
    'booking_changes', 'hotel', 'previous_cancellations', 'has_agent'
]
categorical_columns_encoded = [
    'deposit_type_Non Refund',
    'country_PRT',
    'market_segment_Groups',
    'distribution_channel_TA/TO',
    'market_segment_Direct',
    'distribution_channel_Direct',
    'customer_type_Transient',
    'customer_type_Transient-Party',
    'country_FRA',
    'country_GBR',
    'country_DEU'
]

X_columns = numerical_columns + categorical_columns_encoded
X = df[X_columns]
y = df['is_canceled']

# ── 7. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 8. TRAIN RANDOM FOREST (best params from notebook) ───────────────────────
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    max_features='log2',
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Quick sanity check
from sklearn.metrics import accuracy_score, f1_score
y_pred = rf_model.predict(X_test)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")

# ── 9. SAVE ARTIFACTS ─────────────────────────────────────────────────────────
# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the feature column list so app.py can build the input DataFrame correctly
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X_columns, f)

# Save the top-countries list so app.py applies the same grouping
with open('top_countries.pkl', 'wb') as f:
    pickle.dump(list(top_countries), f)

print("\n✅  Saved: model.pkl, model_columns.pkl, top_countries.pkl")
print("All three files are needed by app.py.")
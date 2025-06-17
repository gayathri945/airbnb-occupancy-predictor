import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ✨ Custom Airbnb-inspired theme with gradient background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #4b79a1 100%);
        color: #e0e0e0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ff5a5f !important;
    }
    p, label, .stSelectbox label, .stNumberInput label {
        color: #f1f1f1 !important;
    }
    .stSidebar {
        background: linear-gradient(180deg, #4b79a1 0%, #2c3e50 100%);
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: #ff5a5f !important;
    }
    .stButton>button {
        background-color: #ff5a5f;
        color: #ffffff;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.4em 1em;
    }
    .stButton>button:hover {
        background-color: #e0484d;
    }
    .stSelectbox, .stNumberInput input {
        background-color: #4a4a4a;
        color: #ffffff;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🏠🌃 Title + refined tagline
st.markdown("<h1 style='text-align:center;'>🏠🏙️ Airbnb Occupancy Rate Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Model and predict Airbnb occupancy rates with real NYC data. "
    "Explore performance and generate personalized predictions.</p>",
    unsafe_allow_html=True
)

# Load data
df = pd.read_csv("listings.csv")
df['occupancy_rate'] = (df['number_of_reviews'] / df['availability_365']) * 100
df['occupancy_rate'] = df['occupancy_rate'].clip(upper=100)
df = df.dropna(subset=['occupancy_rate'])
df = df[(df['availability_365'] > 0) & (df['number_of_reviews'] > 0)]

X = df[['price', 'minimum_nights', 'availability_365']]
y = df['occupancy_rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar model selection
st.sidebar.markdown(
    "<h3 style='color:#ff5a5f; font-size: 22px;'>⚙️ Model Settings</h3>",
    unsafe_allow_html=True
)

model_type = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Decision Tree"])

# Model selection + training
model = LinearRegression() if model_type == "Linear Regression" else DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show model performance
if st.button("✨ Show Model Performance"):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.success(f"📈 *Model RMSE:* {rmse:.2f}%")
    st.caption("📌 RMSE = avg error in % occupancy rate. Lower = better.")

    if model_type == "Decision Tree":
        st.subheader("🔑 Feature Importance")
        for feat, imp in zip(X.columns, model.feature_importances_):
            st.write(f"• *{feat}*: {imp:.2f}")
        st.caption("📌 Feature importance is only shown for Decision Trees as they compute it natively.")

    st.subheader("🎨 Actual vs Predicted Occupancy Rate")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color="#00a699", label="Predictions")
    plt.plot([0, 100], [0, 100], color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual Occupancy Rate (%)", fontsize=12, color="#ffffff")
    plt.ylabel("Predicted Occupancy Rate (%)", fontsize=12, color="#ffffff")
    plt.title("Actual vs Predicted Occupancy Rate", fontsize=14, color="#ffffff")
    plt.grid(True, color="#555555")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    st.pyplot(plt)

# Custom prediction form
st.subheader("🔮 Predict Occupancy Rate for New Listing")
price = st.number_input("Price ($)", 10, 10000, 100)
min_nights = st.number_input("Minimum Nights", 1, 365, 2)
avail = st.number_input("Availability (days/year)", 1, 366, 180)
if st.button("Predict"):
    input_df = pd.DataFrame([[price, min_nights, avail]], columns=X.columns)
    pred = model.predict(input_df)[0]
    st.success(f"✅ Predicted Occupancy Rate: {min(pred, 100):.2f}%")
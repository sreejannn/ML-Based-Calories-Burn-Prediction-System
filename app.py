import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Calories Burn Prediction",
    page_icon="🏋️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

/* Hide Streamlit Default UI */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Main Container */
.main-container {

    width: 65%;

    margin: auto;

    margin-top: 40px;

    padding: 35px;

    border-radius: 20px;

    background: rgba(255,255,255,0.05);

    backdrop-filter: blur(12px);

    box-shadow:
        0px 0px 25px rgba(255,255,255,0.05);
}

/* Metric Card */
[data-testid="stMetric"] {

    background: rgba(255,255,255,0.05);

    padding: 25px;

    border-radius: 20px;

    text-align: center;

    box-shadow:
        0px 0px 20px rgba(0,191,255,0.15);
}

/* Loading Overlay */
.loading-overlay {

    position: fixed;

    top: 0;
    left: 0;

    width: 100%;
    height: 100%;

    background: rgba(0,0,0,0.45);

    backdrop-filter: blur(5px);

    z-index: 999999;

    display: flex;

    justify-content: center;

    align-items: center;
}

/* Spinner */
.loader {

    border: 8px solid rgba(255,255,255,0.15);

    border-top: 8px solid #00BFFF;

    border-radius: 50%;

    width: 80px;

    height: 80px;

    animation: spin 1s linear infinite;
}

/* Spinner Animation */
@keyframes spin {

    100% {
        transform: rotate(360deg);
    }
}

/* Button Styling */
.stButton > button {

    width: 100%;

    background-color: #00BFFF;

    color: white;

    border: none;

    border-radius: 10px;

    height: 50px;

    font-size: 18px;

    font-weight: bold;
}

/* Labels */
label {

    font-size: 16px !important;

    font-weight: 500 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():

    calories = pd.read_csv("calories.csv")

    exercise = pd.read_csv("exercise.csv")

    # Merge datasets
    exercise_df = exercise.merge(
        calories,
        on="User_ID"
    )

    # Remove duplicates
    exercise_df.drop_duplicates(
        subset=["User_ID"],
        keep="last",
        inplace=True
    )

    # Drop User_ID
    exercise_df.drop(
        columns="User_ID",
        inplace=True
    )

    # Encode Gender
    exercise_df = pd.get_dummies(
        exercise_df,
        columns=["Gender"],
        drop_first=True
    )

    exercise_df.rename(
        columns={"Gender_male": "Gender"},
        inplace=True
    )

    # Create BMI Feature
    exercise_df["BMI"] = (
        exercise_df["Weight"] /
        ((exercise_df["Height"] / 100) ** 2)
    )

    exercise_df["BMI"] = round(
        exercise_df["BMI"],
        2
    )

    return exercise_df

# Load dataset
exercise_df = load_data()

# ==========================================
# FEATURE SELECTION
# ==========================================
selected_features = [
    "Age",
    "BMI",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "Gender",
    "Calories"
]

exercise_df = exercise_df[selected_features]

# ==========================================
# SPLIT DATA
# ==========================================
X = exercise_df.drop(
    "Calories",
    axis=1
)

y = exercise_df["Calories"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODELS
# ==========================================
linreg = LinearRegression()

random_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

linreg.fit(X_train, y_train)

random_reg.fit(X_train, y_train)

# ==========================================
# TITLE
# ==========================================
st.title("🏋️ Calories Burn Prediction Dashboard")

st.write(
    """
    Machine Learning-powered fitness tracker
    built using Python, Streamlit, and scikit-learn.
    """
)

# ==========================================
# MAIN CONTAINER
# ==========================================
st.markdown(
    '<div class="main-container">',
    unsafe_allow_html=True
)

# ==========================================
# INPUT FORM
# ==========================================
st.subheader("Enter Your Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # LEFT COLUMN
    with col1:

        age = st.slider(
            "Age",
            18,
            80,
            25
        )

        weight = st.number_input(
            "Weight (kg)",
            30.0,
            150.0,
            70.0
        )

        duration = st.slider(
            "Exercise Duration (minutes)",
            5,
            120,
            30
        )

    # RIGHT COLUMN
    with col2:

        height = st.number_input(
            "Height (cm)",
            120.0,
            220.0,
            170.0
        )

        heart_rate = st.slider(
            "Heart Rate",
            60,
            200,
            100
        )

        body_temp = st.number_input(
            "Body Temperature (°C)",
            35.0,
            42.0,
            37.0
        )

    # Gender
    gender = st.radio(
        "Gender",
        ["Female", "Male"],
        horizontal=True
    )

    # BMI Calculation
    bmi = round(
        weight / ((height / 100) ** 2),
        2
    )

    st.info(f"Calculated BMI: {bmi}")

    # Submit Button
    predict_btn = st.form_submit_button(
        "Predict Calories Burned"
    )

# ==========================================
# PREDICTION SECTION
# ==========================================
if predict_btn:

    # Loading Overlay
    loader = st.empty()

    loader.markdown(
        """
        <div class="loading-overlay">
            <div class="loader"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Fake Delay
    time.sleep(2)

    # Encode Gender
    gender_encoded = 1 if gender == "Male" else 0

    # User Input
    user_data = np.array([[
        age,
        bmi,
        duration,
        heart_rate,
        body_temp,
        gender_encoded
    ]])

    # Prediction
    rf_prediction = random_reg.predict(user_data)

    # Remove Loader
    loader.empty()

    # ==========================================
    # PREDICTION OUTPUT
    # ==========================================
    st.markdown("## 🔥 Estimated Calories Burned")

    st.metric(
        label="Calories Burned",
        value=f"{round(rf_prediction[0], 2)} kcal"
    )

    st.caption("Random Forest Prediction")

    # ==========================================
    # ENTERED DETAILS
    # ==========================================
    st.subheader("Entered Details")

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.write(f"Age: {age}")

        st.write(f"Weight: {weight} kg")

        st.write(f"Height: {height} cm")

        st.write(f"Exercise Duration: {duration} min")

    with detail_col2:

        st.write(f"Heart Rate: {heart_rate} bpm")

        st.write(f"Body Temperature: {body_temp} °C")

        st.write(f"Gender: {gender}")

        st.write(f"BMI: {bmi}")

# ==========================================
# CLOSE CONTAINER
# ==========================================
st.markdown(
    "</div>",
    unsafe_allow_html=True
)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.caption(
    "💡 Developed by Sreejan Narapareddy | Machine Learning-Powered Fitness Tracker"
)
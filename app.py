
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

from recommendation import get_recommendations


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "student_model.pkl"
DATA_FILE = BASE_DIR / "data" / "student_data.csv"


# Streamlit configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# Load trained model
@st.cache_resource
def load_model():
    with MODEL_FILE.open("rb") as file:
        return pickle.load(file)


# Application title
st.title("🎓 Student Performance Prediction System")

st.write(
    "Predict academic performance and receive personalized "
    "learning recommendations."
)

st.info(
    "This demonstration uses synthetic sample data. "
    "Predictions are estimates, not guaranteed exam results."
)


# Check model file
if not MODEL_FILE.exists():
    st.error(
        "Trained model not found. "
        "Run generate_dataset.py and train_model.py first."
    )
    st.stop()


# Load model safely
try:
    model_data = load_model()
    model = model_data["model"]
    features = model_data["features"]

except (OSError, pickle.UnpicklingError, EOFError, KeyError) as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# Student input form
st.subheader("Enter Student Details")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

    with col2:
        assignment_score = st.number_input(
            "Assignment Score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        previous_score = st.number_input(
            "Previous Exam Score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "Predict Performance",
        type="primary",
        use_container_width=True
    )


# Make prediction
if submitted:

    # Match input values to the trained model's feature names
    input_values = {
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Assignment_Score": assignment_score,
        "Previous_Score": previous_score,
        "attendance": attendance,
        "study_hours": study_hours,
        "assignment_score": assignment_score,
        "previous_score": previous_score
    }

    missing_features = [
        feature for feature in features
        if feature not in input_values
    ]

    if missing_features:
        st.error(
            f"Unsupported model features: {missing_features}. "
            "Check train_model.py and retrain the model if needed."
        )
        st.stop()

    student_data = pd.DataFrame(
        [[input_values[feature] for feature in features]],
        columns=features
    )

    try:
        predicted_score = float(model.predict(student_data)[0])

    except (ValueError, TypeError) as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    predicted_score = max(0.0, min(100.0, predicted_score))

    # Performance category
    if predicted_score < 40:
        category = "Needs Improvement"
    elif predicted_score < 70:
        category = "Average"
    else:
        category = "Good"

    # Display results
    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Final Score",
            f"{predicted_score:.1f} / 100"
        )

    with col2:
        st.metric(
            "Performance Category",
            category
        )

    st.progress(int(round(predicted_score)))

    # Recommendations
    st.subheader("Personalized Learning Recommendations")

    recommendations = get_recommendations(
        attendance=attendance,
        study_hours=study_hours,
        assignment_score=assignment_score,
        previous_score=previous_score,
        predicted_score=predicted_score
    )

    if recommendations:
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")
    else:
        st.success("Keep following your current study routine!")


# Dataset analysis
st.divider()
st.subheader("Dataset Analysis")

if DATA_FILE.exists():

    try:
        data = pd.read_csv(DATA_FILE)

        required_columns = [
            "attendance",
            "study_hours",
            "final_score"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            st.error(
                f"Dataset columns missing: {missing_columns}"
            )
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.write("First 10 Student Records")
            st.dataframe(
                data.head(10),
                use_container_width=True
            )

        with col2:
            st.write("Study Hours vs Final Score")

            st.scatter_chart(
                data,
                x="study_hours",
                y="final_score"
            )

        # Average score by attendance range
        st.write("Average Score by Attendance Range")

        data["attendance_range"] = pd.cut(
            data["attendance"],
            bins=[-0.001, 50, 60, 70, 80, 90, 100],
            labels=[
                "0-50",
                "50-60",
                "60-70",
                "70-80",
                "80-90",
                "90-100"
            ],
            include_lowest=True
        )

        average_scores = (
            data.groupby(
                "attendance_range",
                observed=True
            )["final_score"]
            .mean()
            .dropna()
        )

        if not average_scores.empty:
            average_scores.index = average_scores.index.astype(str)

            st.bar_chart(average_scores)
        else:
            st.warning(
                "No valid attendance data available for the chart."
            )

    except (OSError, ValueError, KeyError) as error:
        st.error(f"Dataset analysis failed: {error}")

else:
    st.warning(
        "Dataset not found. Run generate_dataset.py first."
    )


# Sidebar model information
st.sidebar.title("Model Information")
st.sidebar.write("Algorithm: Random Forest Regressor")

st.sidebar.metric(
    "Test MAE",
    f"{model_data['mae']:.2f}"
)

st.sidebar.metric(
    "Test R² Score",
    f"{model_data['r2']:.3f}"
)

st.sidebar.caption(
    "Evaluation metrics are based on synthetic sample data."
)

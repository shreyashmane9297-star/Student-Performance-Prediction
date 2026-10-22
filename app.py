import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load trained model
try:
    model = joblib.load("student_performance_model.pkl")
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

# Application title
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's final score using Machine Learning.")

st.info("Passing threshold: 40/100")

st.divider()

# Student inputs
st.subheader("📝 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=15.0,
        value=6.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

with col2:
    previous_score = st.number_input(
        "Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    assignment_score = st.number_input(
        "Assignment Score",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

with col3:
    sleep_hours = st.number_input(
        "Sleep Hours Per Day",
        min_value=0.0,
        max_value=15.0,
        value=8.0,
        step=0.5
    )

st.divider()

# Prediction button
if st.button("🔮 Predict Student Result", use_container_width=True):

    student_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_score": [previous_score],
        "assignment_score": [assignment_score],
        "sleep_hours": [sleep_hours]
    })

    try:
        prediction = float(model.predict(student_data)[0])
        prediction = max(0.0, min(100.0, prediction))

        st.subheader("📊 Prediction Results")

        score_col, result_col = st.columns(2)

        with score_col:
            st.metric(
                "Predicted Final Score",
                f"{prediction:.2f} / 100"
            )
            st.progress(prediction / 100)

        with result_col:
            if prediction >= 40:
                st.success("✅ PREDICTED RESULT: PASS")
            else:
                st.error("❌ PREDICTED RESULT: FAIL")

        st.caption(
            "PASS: predicted score is at least 40. "
            "FAIL: predicted score is below 40."
        )

        with st.expander("View Student Details"):
            st.dataframe(student_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

with st.expander("📌 About This Project"):
    st.write(
        "This application uses a Random Forest Regression model "
        "to estimate a student's final examination score."
    )
    st.write("Dataset: 2,005 synthetic student records.")
    st.warning(
        "This is an educational demonstration. Predictions are "
        "estimates and are not official examination results."
    )
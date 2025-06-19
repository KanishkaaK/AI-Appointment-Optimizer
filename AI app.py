import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model and label encoder
model = joblib.load('no_show_model.joblib')
le_appointment_type = joblib.load('label_encoder.joblib')
le_doctor = joblib.load('label_encoder_doctor.joblib')  # New encoder for doctor

# Title
st.title("AI Appointment Predictor")

# Helper function: convert to 12hr format
def hour_12_format(hour):
    am_pm = "AM" if hour < 12 else "PM"
    hour_display = hour % 12
    hour_display = 12 if hour_display == 0 else hour_display
    return f"{hour_display}:00 {am_pm}"

# Doctor dropdown
doctor_list = le_doctor.classes_
selected_doctor = st.selectbox("Select Doctor", doctor_list)
doctor_encoded = le_doctor.transform([selected_doctor])[0]

# 12-hour format dropdown
appointment_hour_12 = st.selectbox(
    "Select Appointment Time (12-Hour Format)",
    options=range(0, 24),
    format_func=hour_12_format
)

# Day of Week
appointment_day_of_week = st.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x]
)

# Expected Delay
delay_mins = st.number_input("Expected Delay (mins)", min_value=0, value=0)

# Appointment Type
appointment_type = st.selectbox("Appointment Type", le_appointment_type.classes_)
appointment_type_encoded = le_appointment_type.transform([appointment_type])[0]

# Predict button
if st.button("Predict Probability"):
    # Prepare input
    input_data = pd.DataFrame(
        [[doctor_encoded, appointment_hour_12, appointment_day_of_week, delay_mins, appointment_type_encoded]],
        columns=['doctor_id_encoded', 'appointment_hour', 'appointment_day_of_week', 'delay_mins', 'appointment_type_encoded']
    )

    # Predict probability 
    prob_no_show = model.predict_proba(input_data)[0, 1]

    # Display
    st.write(f"### 🎯 Predicted  Probability: **{prob_no_show:.2f}**")

    # Color bar
    st.progress(1 - prob_no_show)

    # Available or Not Available + Suggestion
    if prob_no_show < 0.3:
        st.success(" Available - Appointment is likely to be attended.")
        suggestion = "No action needed."
    elif 0.3 <= prob_no_show < 0.6:
        st.warning(" Medium Risk - Recommend confirming with the client.")
        suggestion = "Recommend phone confirmation."
    else:
        st.error(" Not Available - High risk of no-show.")
        suggestion = "Consider rescheduling or double-check with client."

    # Log this prediction to a CSV
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'doctor_id': selected_doctor,
        'appointment_hour_12': hour_12_format(appointment_hour_12),
        'appointment_day_of_week': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][appointment_day_of_week],
        'delay_mins': delay_mins,
        'appointment_type': appointment_type,
        'no_show_probability': round(prob_no_show, 2),
        'suggestion': suggestion
    }

    log_df = pd.DataFrame([log_entry])

    # Append or create log file
    if os.path.exists("prediction_log.csv"):
        log_df.to_csv("prediction_log.csv", mode='a', header=False, index=False)
    else:
        log_df.to_csv("prediction_log.csv", index=False)

    st.info("Prediction logged ")

# Option to download log file
if os.path.exists("prediction_log.csv"):
    with open("prediction_log.csv", "rb") as f:
        st.download_button(" Download Prediction Log (CSV)", f, file_name="prediction_log.csv")

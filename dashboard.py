import streamlit as st
import pickle
import numpy as np
from phase5 import Perceptron

# 1. Load the model
with open('student_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Student Performance Predictor")
st.write("Predicting if a student will Pass or Fail based on the Perceptron model.")

# 2. Input Sliders
studytime = st.slider("Study Time (1=Low, 4=High)", min_value=1, max_value=4, value=2)
absences = st.slider("Number of Absences", min_value=0, max_value=34, value=5)
g1_grade = st.slider("Previous Grade (G1)", min_value=0, max_value=20, value=10)

# 3. Prediction Button
if st.button("Predict Final Status"):
    # Create a placeholder array of 32 zeros (matching your training data shape)
    full_input = np.zeros(32)

    # Fill in the specific indexes for your features
    # (Double check these indexes match your X_train columns!)
    full_input[13] = studytime  # Index for 'studytime'
    full_input[29] = absences   # Index for 'absences'
    full_input[30] = g1_grade   # Index for 'G1'

    # Reshape to (1, 32)
    student_data = full_input.reshape(1, -1)

    # Make the prediction
    prediction = model.predict(student_data)[0]
    
    # 4. Display the Results
    # Since Perceptron is a binary classifier (0 or 1), 
    # we display the outcome rather than a numeric grade.
    if prediction == 1:
        st.success("✅ Prediction: PASS")
        st.balloons()
    else:
        st.error("🚨 Prediction: FAIL")
        st.write("Intervention recommended: This student is at risk.")
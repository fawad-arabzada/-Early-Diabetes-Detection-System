import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
from my_code import model, scaler  

st.set_page_config(page_title="Diabetes Detection System", page_icon=":guardsman:", layout="wide")

st.title("Early Diabetes Detection Made System")
st.markdown("Explore our Diabetes Detection System to predict your risk of diabetes and get health advice.")

image_url = "https://images.unsplash.com/photo-1685660478008-69441afef904?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OTJ8fGRpYWJldGVzfGVufDB8fDB8fHww"
st.markdown(f"""
    <div style="float:left; margin-right:20px;" class="image-container">
        <img src="{image_url}" width="300" style="border: 2px solid #ccc; border-radius: 10px;"/>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["🏠 Home", "🔮 Prediction", "ℹ️ About"])

if page == "🏠 Home":
    st.markdown("Welcome to the Diabetes Detection System. This system helps you predict your risk of diabetes based on health data. Enter your health data to check your risk.")
    if st.button("Check Your Risk"):
        st.session_state.page = "🔮 Prediction"
        st.rerun() 

elif page == "🔮 Prediction":
    st.subheader("Enter Your Health Details")

    with st.form("user_input_form"):
        pregnancies = st.text_input("Number of Pregnancies", "0")
        glucose = st.text_input("Glucose Level (mg/dL)", "100")
        blood_pressure = st.text_input("Blood Pressure (mm Hg)", "80")
        skin_thickness = st.text_input("Skin Thickness (mm)", "20")
        insulin = st.text_input("Insulin Level (μU/mL)", "80")
        bmi = st.text_input("Body Mass Index (BMI)", "25.0")
        diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function", "0.5")
        age = st.text_input("Age", "30")

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            user_data = np.array([[
                float(pregnancies), float(blood_pressure), float(skin_thickness),
                float(insulin), float(bmi), float(age), float(glucose), 
                float(diabetes_pedigree_function)
            ]])

            user_data_scaled = scaler.transform(user_data)

            prediction = model.predict(user_data_scaled)[0]
            probability = model.predict_proba(user_data_scaled)[0][1]

            result_text = "Yes" if prediction == 1 else "No"
            st.write(f"Diabetes Prediction: {result_text}")
            st.write(f"Prediction Probability: {probability:.2f}")



            recommendation = "Consult a doctor" if prediction == 1 else "Maintain a healthy lifestyle"
            if prediction == 1:
                st.warning("High risk detected. Please consult a doctor.")
            else:
                st.success("Low risk. Keep up the good habits!")

            report = FPDF()
            report.add_page()
            report.set_font("Arial", size=12)
            report.cell(200, 10, txt="Diabetes Prediction Report", ln=True, align="C")
            for field, value in zip(
                ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "Glucose", "DiabetesPedigreeFunction"],
                user_data[0]
            ):
                report.cell(200, 10, txt=f"{field}: {value}", ln=True)
            report.cell(200, 10, txt=f"Prediction: {result_text}", ln=True)
            report.cell(200, 10, txt=f"Likelihood: {probability:.2f}", ln=True)
            report.cell(200, 10, txt=f"Recommendation: {recommendation}", ln=True)
            report.output("diabetes_prediction_report.pdf")

            with open("diabetes_prediction_report.pdf", "rb") as file:
                st.download_button("Download Report", file, "Diabetes_Prediction_Report.pdf")

        except Exception as e:
            st.error(f"Error in prediction: {e}")

elif page == "ℹ️ About":
    st.subheader("About the System")
    st.markdown("""
    **Diabetes Detection System** is a machine learning-based application that helps in early 
    detection of diabetes using health parameters such as blood pressure, glucose level, BMI, 
    insulin level, and more. It is powered by a **Logistic Regression model**, trained using 
    the **Pima Indian Diabetes dataset**.

    **Features of this system:**
    - Uses **machine learning** for diabetes prediction.
    - Accepts **user health details** for prediction.
    - Provides **probability-based results**.
    - Generates a **PDF report** with recommendations.
    - Easy to use with a simple **Streamlit interface**.
    """)

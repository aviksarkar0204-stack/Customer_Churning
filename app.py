import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")
st.markdown("Upload a CSV file of customer data to predict churn.")

model = joblib.load('Logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

uploaded_file = st.file_uploader("Upload customer CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    try:
        scaled = scaler.transform(df)
        predictions = model.predict(scaled)
        probabilities = model.predict_proba(scaled)[:, 1]

        df['Churn_Prediction'] = predictions
        df['Churn_Probability'] = probabilities.round(2)
        df['Churn_Prediction'] = df['Churn_Prediction'].map({1: 'Yes', 0: 'No'})

        st.subheader("Prediction Results")
        st.dataframe(df[['Churn_Prediction', 'Churn_Probability']])

        churn_count = (df['Churn_Prediction'] == 'Yes').sum()
        total = len(df)
        st.metric("Total Customers", total)
        st.metric("Predicted to Churn", churn_count)
        st.metric("Churn Rate", f"{(churn_count/total*100):.1f}%")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Make sure your CSV has the same columns as the training data.")
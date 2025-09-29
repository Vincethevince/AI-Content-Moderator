import streamlit as st
import requests

st.title("AI Content Moderator")
text = st.text_area("Enter a comment:")
url = "https://ai-content-moderator-822949816423.europe-west1.run.app/predict"
if st.button("Predict"):
    response = requests.post(url, json={"text": text}).json()
    st.write(response["response"])
import re
import streamlit as st
from banking_rag import get_rag_answer
from banking_intent import predict_intent

st.set_page_config(page_title="SBI Banking Assistant", page_icon="🏦", layout="centered")
st.markdown(
    """
    <style>
    /* Page background and font */
    .stApp {
        background-color: #dce9f5;
        font-family: 'Arial', sans-serif;
    }
    /* Force heading color */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h4 {
        color: #002663 !important; /* SBI dark blue */
        font-weight: 700 !important;
    }
    /* General text color */
    p, div, label, span {
        color: #002663 !important;
    }
    /* Input box */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #002663;
        border: 1px solid #6c757d;
        padding: 10px;
        border-radius: 8px;
    }
    /* Button */
    .stButton>button {
        background-color: #0057b7;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
        border: none;
    }
    /* Hide default footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1 style='text-align: center;'>🏦SBI Banking Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; '>Your trusted AI partner for SBI banking queries</h4>", unsafe_allow_html=True)
st.write("---")

user_input = st.text_input("Enter your query:")
if user_input:
    with st.spinner("Generating answer..."):
        label, confidence = predict_intent(user_input, temperature=0.1)
        if label == 'banking':
            st.success(label)
            response = get_rag_answer(user_input)
            response = re.sub(r"<think>.*?</think>\n\n?", "", response, flags=re.DOTALL)
            st.success("Here's the information you requested:")
            st.write(response)
        else:
            st.error(label)
            st.error("Sorry, I can assist only with banking-related queries. Please ask a banking-related question.")
            
st.markdown(
    """
    <hr style='border: 0.5px solid #6c757d;'>
    <div style='text-align: center; color: #002663;'>
      Powered by SBI AI Assistant
    </div>
    """,
    unsafe_allow_html=True,
)

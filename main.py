import pandas as pd
import streamlit as st
from screen_tiabs import screen_me
from evaluate import evaluate_me

def get_data():
    st.session_state.llm = st.selectbox("LLM", ["gpt-4o-mini", "gpt-o3", "gpt-4o", "gpt-4.1",
                                                "Other OpenAI model or specific snapshot"])

    if st.session_state.llm == "Other OpenAI model or specific snapshot":
        st.session_state.llm = st.text_input("Enter model name or snapshot, eg. o4-mini-2025-04-16")

    st.session_state.key = st.text_input("Paste your OpenAI API key here", type="password")

def init_session():
    if "task" not in st.session_state:
        st.session_state.task=""
    if "llm" not in st.session_state:
        st.session_state.llm=""
    if 'results_df' not in st.session_state:
        st.session_state.results_df=pd.DataFrame()

init_session()

st.header("Automating Evidence Synthesis and Related Tasks with LLMs")
st.write("A UI to Lena's collection of scripts to interact with OpenAI LLMs - users are responsible for their own prompt development and evaluation and need to ensure that they follow ethical and scientific principles. Minimum best practices of reporting include: reporting the use of LLMs, LLM name, LLM version, LLM prompts, description of prompt development process, and if applicable: full confusion matrix (True Positives, True negatives, False positives, False negatives) plus precision, recall, for easy overview of performances.")

st.session_state.task = st.selectbox("Automation Task", ["Screening", "Evaluation", "Data Extraction", "Fulltext interrogation", "PhD Chapter", "Synthetic Abstract"], placeholder="Select Task")


if st.session_state.task:
    if st.session_state.task == "Screening":
        get_data()
        if st.session_state.llm and st.session_state.key:
            screen_me()
    elif st.session_state.task == "Evaluation":
        evaluate_me()
    else:
        st.write("Task not yet implemented")



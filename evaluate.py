import streamlit as st
import chardet
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def detect_encoding(file):
    raw_data = file.read()
    file.seek(0)  # Reset file pointer after reading
    result = chardet.detect(raw_data)
    return result['encoding'], result['confidence']

def evaluate_me():
    st.header("Evaluate predictions")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Detect encoding
        encoding, confidence = detect_encoding(uploaded_file)
        encoding_var = encoding  # Store in variable for later use

        st.write(f"**Detected Encoding:** {encoding_var} (Confidence: {confidence:.2f})")

        df = pd.read_csv(uploaded_file, encoding=encoding_var)
        st.success("CSV loaded successfully!")
        st.markdown("### Data overview:")
        st.dataframe(df.head())

        st.markdown("### Gold standard column where 1 refers to included or relevant data, and 0 to excluded or irrelevant data.")
        opts = [""]
        opts.extend(list(df.columns))
        gold_column = st.selectbox("Select gold standard", opts, index=0)

        st.markdown("### AI prediction column where 1 refers to included or relevant data, and 0 to excluded or irrelevant data.")
        pred_column = st.selectbox("Select predictions", opts, index=0)

        if len(gold_column) > 0 and len(pred_column) > 0 and gold_column!=pred_column:
            gold = [int(g) for g in df[gold_column]]
            pred = [int(g) for g in df[pred_column]]

            st.write("Sample data from gold and pred column:")
            st.write(gold[:5])
            st.write(pred[:5])

            st.markdown("## Results")
            st.write("Recall (sensitivity): {}".format(
                recall_score(gold, list(st.session_state.results_df["LLM prediction"]))))
            st.write("Precision (positive-predictive-value): {}".format(
                precision_score(gold, list(st.session_state.results_df["LLM prediction"]))))

            st.write("Confusion Matrix:")
            disp = ConfusionMatrixDisplay(
                confusion_matrix=confusion_matrix(gold, list(st.session_state.results_df["LLM prediction"])))
            disp.plot()
            st.pyplot(plt)
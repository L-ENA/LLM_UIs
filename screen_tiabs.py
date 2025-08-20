import streamlit as st
import streamlit as st
import pandas as pd
import chardet
import pyperclip
import openai
from openai import OpenAI
from datetime import date

def detect_encoding(file):
    raw_data = file.read()
    file.seek(0)  # Reset file pointer after reading
    result = chardet.detect(raw_data)
    return result['encoding'], result['confidence']

def screen_me():
    with st.sidebar:
        if st.button("Reset Results"):
            st.session_state.results_df=pd.DataFrame()
        st.session_state.minlength=st.number_input("Minimum length of context, shorter inputs are classified as includes automatically", value=200)

    st.subheader("LLM Screening")
    st.write("For screening and data extraction, it is important to use random independent subsets of data for developing and validating prompts. Do this with Excel, Python/R, other spreadsheet software, or use [RefRandomiser](https://refrandomiser.streamlit.app/) to quickly and easily create as many subsets as needed.")

    st.markdown("### Upload CSV")


    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Detect encoding
        encoding, confidence = detect_encoding(uploaded_file)
        encoding_var = encoding  # Store in variable for later use

        st.write(f"**Detected Encoding:** {encoding_var} (Confidence: {confidence:.2f})")


        df = pd.read_csv(uploaded_file, encoding=encoding_var)
        st.success("CSV loaded successfully!")
        st.markdown("### Data overview:")
        st.dataframe(df.style.set_properties(**{'background-color': 'white'}).head())

        st.markdown("### Select text to classify")
        st.write("Please select columns that contain relevant text. For running the LLM on titles and abstracts, first select the title column and then select the abstract column. The order matters!")
        my_selections=st.multiselect("Select text data", list(df.columns))
        st.markdown("#### Example context (first row of spreadsheet)")
        st.write("Check if selected order and fields make sense as LLM context. The LLM will automatially assign 'included' status if the given context is shorter than 200 characters")
        with st.expander("Show example from spreadsheet"):
            for s in my_selections:
                st.markdown("**{}**: {}".format(s, df[s][0]))
        #st.write(my_selections)

        if len(my_selections)>0:
            st.markdown("### Prompt instructions")

            # c1, c2= st.columns([6,1])
            # with c1:
            #     st.write("Note, for automated processing with this script, the prompt should always finish with the following sentence:")
            #     prompt_end= st.text_area("Answer YES if the article is relevant or unclear. Answer NO if it is not. Then reproduce the exact context from the paper that contained the information on which basis you made the decision. Here is the text of the article: ")
            #
            # with c2:
            #     if st.button("Copy"):
            #         pyperclip.copy(prompt_end)
            # st.divider()


            st.markdown("""
                    Prompting instructions. Remember to:
                    - set the scene (ie. tell the LLM its role)
                    - describe the review type and study type you are looking for
                    - describe inclusion criteria
                    - optionally, describe exclusion criteria
                    - use text from example below to make sure that screening decisions get parsed correctly
                """)
            st.write("Below is an example prompt. You can copy and edit it, however, retaining the last sentences (after 'Answer YES if the article is relevant or unclear [..]' is recommended. You do not need to paste any actual context, this will be added automatically from your spreadsheet.")
            st.divider()
            c1full, c2full = st.columns([6, 1])
            # with c1full:
            prompt_example = "You are a researcher screening published journal articles for inclusion in a literature analysis. The inclusion criteria are the following: A cohort or longitudinal study. The study should involve participants of any age. The research should focus on the effect or association of any dietary pattern on the incidence of depression or anxiety. Answer YES if the article is relevant or unclear. Answer NO if it is not. Then reproduce the exact context from the paper that contained the information on which basis you made the decision. Here is the text of the article: "
            st.write(prompt_example)
            # with c2full:
                # if st.button("Copy"):
                #     pyperclip.copy(prompt_example)

            st.markdown("### Enter your prompt:")
            st.text_area("Full prompt", key="llm_prompt")

            if st.session_state.llm_prompt:
                if st.button("Go!") and st.session_state.results_df.shape[0]==0:

                    try:
                        client = OpenAI(
                            api_key=st.session_state.key
                        )

                    except Exception as e:
                        # unknown error
                        print("Error authenticating API key")
                        raise e


                    st.info("Your dataset has {} rows and your prompt is: {}".format(df.shape[0], st.session_state.llm_prompt))
                    progress_text = "Prediction in progress. Please wait."
                    my_bar = st.progress(0, text=progress_text)
                    predictions = []
                    justifications = []
                    for i, row in df.iterrows():
                        my_bar.progress(i/df.shape[0], text=progress_text)
                        contexts=[]
                        for s in my_selections:
                            contexts.append(str(row[s]).strip())
                        ti_abs_key=" ".join(contexts)
                        prompt = "{} {}".format(st.session_state.llm_prompt,ti_abs_key)
                        completion = client.chat.completions.create(
                            model=st.session_state.llm,
                            messages=[
                                {"role": "user", "content": '%s' % prompt}
                            ]
                        )
                        st.session_state.my_model= completion.model
                        openai_response = completion.choices[0].message.content
                        if openai_response.startswith("YES") or openai_response.startswith(
                                "**YES**") or "YES" in openai_response[:15] or len(ti_abs_key) < st.session_state.minlength:
                            predictions.append(1)
                        else:
                            predictions.append(0)
                        justifications.append(openai_response.replace("\n", " ").replace("  ", " "))
                    my_bar.progress(100, text=progress_text)
                    df["LLM prediction"] = predictions
                    df["LLM Justification"] = justifications
                    st.session_state.results_df=df

            if st.session_state.results_df.shape[0]>0:
                st.markdown("### Results")
                st.write("Your predictions were added to the spreadsheets, in new columns called 'LLM prediction' and 'LLM Justification'. You can download results as CSV by hovering and clicking on the top-right of the table below.")
                st.dataframe(st.session_state.results_df)

                st.markdown("### Optional: Eval")
                st.write("Select a column in your spreadsheet that contains gold-standard labelled data - where values are either 1 (include or relevant record) or 0 (exclude or irrelevant). This script will then calculate recall (sensitivity), precision (positive-predictive-value) and other important metrics. Note, specificity may not very meaningful in scenarios with big imbalances between in/ and exclude ratio.")
                opts=[""]
                opts.extend(list(st.session_state.results_df.columns))
                gold_column=st.selectbox("Select gold standard", opts, index=0)

                if len(gold_column)>0:
                    gold=[int(g) for g in df[gold_column]]

                    st.write("Sample data from gold column:")
                    st.write(gold[:5])
                    from sklearn.metrics import confusion_matrix,recall_score, precision_score, ConfusionMatrixDisplay
                    import matplotlib.pyplot as plt
                    st.write("Recall (sensitivity): {}".format(recall_score(gold, list(st.session_state.results_df["LLM prediction"]))))
                    st.write("Precision (positive-predictive-value): {}".format(
                        precision_score(gold, list(st.session_state.results_df["LLM prediction"]))))

                    st.write("Confusion Matrix:")
                    disp=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(gold, list(st.session_state.results_df["LLM prediction"])))
                    disp.plot()
                    st.pyplot(plt)

                    st.markdown("### Reporting the use of AI")
                    st.write("The following paragraph shows a suggested description of the methodology followed by this app. You can copy and use it but you may need to fill in some gaps.")

                    prompt_structure=[
                                {"role": "user", "content": '%s' % st.session_state.llm_prompt}
                            ]
                    openai.api_key=st.session_state.key
                    mytext="OpenAI's '{}' model was used on {}. For each record in the dataset, this request/prompt was sent to the OpenAI API to retrieve classifications: '{}'. For each prompt, context from the following fields was provided: {}. Prompts were developed on [XXXX] randomly selected records and validated on an independent subset of [XXXX] records. Recall, precision, as well as the numbers of true positives, true negatives, false positives and false negatives on the independent test set are reported. All contexts shorter than {} characters were automatically assigned the positive class. Code for the interaction with the API and calculation of results is available here: https://github.com/L-ENA/LLM_UIs".format(st.session_state.my_model, date.today().strftime("%Y-%m-%d"),prompt_structure, "+".join(my_selections), st.session_state.minlength)

                    # c1, c2 = st.columns([6, 1])
                    # with c1:
                    st.write(mytext)
                    # with c2:
                    #     if st.button("Copy report draft"):
                    #         pyperclip.copy(prompt_example)





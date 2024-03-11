import streamlit as st
from time import sleep
from stqdm import stqdm  # for getting animation after submit event

from transformers import pipeline, TFAutoModelForQuestionAnswering, AutoTokenizer, TFAutoModelForSequenceClassification
import json
import spacy
import spacy_streamlit


def draw_all(
        key,
        plot=False,
):

    st.write(
        """
        # NLP Web App

        This app harnesses the power of pretrained transformers, 
        capable of performing wonders with textual data. ✨
        
        ```python
        # Key Features:
        1. Advanced Text Summarizer
        2. Named Entity Recognition
        3. Sentiment Analysis
        4. Question Answering


        ```
        """
    )


with st.sidebar:
    draw_all("sidebar")


def main():

    menu = ["--Select--", "Summarizer", "Named Entity Recognition",
            "Sentiment Analysis", "Question Answering"]
    choice = st.sidebar.selectbox("**Explore Your Options!**", menu)

    if choice == "--Select--":
        st.title("NLP Web App")
        st.markdown(f'<h4 style="{"color: #FF4E5B;"}">Unleashing the Power of Language</h4>', unsafe_allow_html=True)

        st.write("""
                 This is a Natural Language Processing Based Web App that enhance your 
                 experience with text.
        """)

        st.write("""
                Natural Language Processing (NLP) is a branch of Artificial Intelligence (AI) focused on enabling 
                machines to understand the human language in the way they spoke and write, interpret, and generate 
                human-like language.
        """)

        # st.write("""
        #          #### How to Use:
        #          1. Select a feature from the sidebar.
        #          2. Input your text in the designated area.
        #          3. Click 'Process' and explore the results.
        #
        # """)
        #
        # st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTOnQDBi_jgep5o9teVj21rjAJaLFA1EDDHg&usqp=CAU'
        #          , use_column_width=True)

        # Create a layout with two columns
        col1, col2 = st.columns([1, 2])

        # Image on the left
        col1.image(
            'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTOnQDBi_jgep5o9teVj21rjAJaLFA1EDDHg&usqp=CAU')

        # How to Use Section on the right
        col2.write("""
            #### How to Use:
            1. Select a feature from the sidebar.
            2. Input your text in the designated area.
            3. Press "Ctrl+Enter" to process.
        """)

    elif choice == "Summarizer":
        st.subheader("Text Summarization 📝")
        st.write(" Enter the Text you want to summarize !")
        raw_text = st.text_area("Enter Your Text Here")
        num_words = st.number_input("Enter Number of Words in Summary", value=0, format="%d")

        if raw_text != "" and num_words is not None:
            num_words = int(num_words)
            summarizer = pipeline('summarization')
            summary = summarizer(raw_text, min_length=num_words, max_length=50)
            s1 = json.dumps(summary[0])
            d2 = json.loads(s1)

            # Change the progress bar message
            for _ in stqdm(range(50), desc="Generating Summary..."):
                sleep(0.1)

            result_summary = d2['summary_text']
            result_summary = '. '.join(list(map(lambda x: x.strip().capitalize(),
                                                result_summary.split('.'))))
            st.write(f"Here's your Summary:")
            st.markdown(f"<p style='color: #008080;'>{result_summary}</p>", unsafe_allow_html=True)

    elif choice == "Named Entity Recognition":
        nlp = spacy.load("en_core_web_sm")
        st.subheader("Named Entity Recognition 🏷️")
        st.write(" Enter the Text below To extract Named Entities !")

        raw_text = st.text_area("Enter Text Here")
        if raw_text.strip() != "":
            doc = nlp(raw_text)
            for _ in stqdm(range(50), desc="Extracting Named Entities..."):
                sleep(0.1)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="List of Entities")

    # elif choice == "Sentiment Analysis":
    #     st.subheader("Sentiment Analysis 🎭")
    #     sentiment_analysis = pipeline("sentiment-analysis")
    #     st.write(" Enter the Text below To find out its Sentiment !")
    #
    #     raw_text = st.text_area("Enter Text Here")
    #     if raw_text.strip() != "":
    #
    #         threshold = 0.7  # Adjust this threshold as needed
    #
    #         result = sentiment_analysis(raw_text)[0]
    #         sentiment = result['label']
    #         score = result['score']
    #
    #         for _ in stqdm(range(50), desc="Analyzing sentiment..."):
    #             sleep(0.1)
    #
    #         if sentiment == "POSITIVE" and score >= threshold:
    #             st.markdown('<h2 style="color: green; font-size: 28px;"> Positive 😊</h2>', unsafe_allow_html=True)
    #         elif sentiment == "NEGATIVE" and score >= threshold:
    #             st.markdown('<h2 style="color: red; font-size: 28px;"> Negative 😞</h2>', unsafe_allow_html=True)
    #         else:
    #             st.markdown('<h2 style="font-size: 28px;"> Neutral 😐</h2>', unsafe_allow_html=True)

    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis 🎭")
        st.write(" Enter the Text below To find out its Sentiment !")

        # Replace 'nlptown/bert-base-multilingual-uncased-sentiment' with the model you want to use
        model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

        # Download the sentiment analysis model
        sentiment_analysis_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use the pipeline with the downloaded model and tokenizer
        sentiment_analysis = pipeline("sentiment-analysis", model=sentiment_analysis_model,
                                      tokenizer=sentiment_analysis_tokenizer)

        raw_text = st.text_area("Enter Text Here")
        if raw_text.strip() != "":
            threshold = 0.7  # Adjust this threshold as needed

            result = sentiment_analysis(raw_text)[0]
            sentiment = result['label']
            score = result['score']

            if sentiment == "POSITIVE" and score >= threshold:
                st.markdown('<h2 style="color: green; font-size: 28px;"> Positive 😊</h2>', unsafe_allow_html=True)
            elif sentiment == "NEGATIVE" and score >= threshold:
                st.markdown('<h2 style="color: red; font-size: 28px;"> Negative 😞</h2>', unsafe_allow_html=True)
            else:
                st.markdown('<h2 style="font-size: 28px;"> Neutral 😐</h2>', unsafe_allow_html=True)

        #     result = sentiment_analysis(raw_text)[0]
        #     sentiment = result['label']
        #     for _ in stqdm(range(50), desc="Analyzing sentiment..."):
        #         sleep(0.1)
        #     if sentiment == "POSITIVE":
        #         st.markdown('<h2 style="color: green; font-size: 28px;"> Positive 😊</h2>', unsafe_allow_html=True)
        #     elif sentiment == "NEGATIVE":
        #         st.markdown('<h2 style="color: red; font-size: 28px;"> Negative 😞</h2>', unsafe_allow_html=True)
        #     elif sentiment == "NEUTRAL":
        #         st.markdown('<h2 style="font-size: 28px;"> Neutral 😐</h2>', unsafe_allow_html=True)

    elif choice == "Question Answering":
        st.subheader("Question Answering 🧩")
        st.write(" Enter the Context and ask the Question to find out the Answer !")

        # Replace 'distilbert-base-cased-distilled-squad' with the model you want to use
        model_name = 'distilbert-base-cased-distilled-squad'

        # Download the model and tokenizer
        question_answering_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        question_answering_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use the pipeline with the downloaded model and tokenizer
        question_answering = pipeline("question-answering", model=question_answering_model,
                                      tokenizer=question_answering_tokenizer)

        context = st.text_area("Enter the Context Here")
        question = st.text_area("Enter your Question Here")

        if context.strip() != "" and question.strip() != "":
            result = question_answering(question=question, context=context)
            generated_text = result['answer']
            generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
            st.write(f"Here's your Answer:")
            st.markdown(f"<p style='color: #008080;'>{generated_text}</p>", unsafe_allow_html=True)

    # elif choice == "Question Answering":
    #     st.subheader("Question Answering 🧩")
    #     st.write(" Enter the Context and ask the Question to find out the Answer !")
    #     question_answering = pipeline("question-answering")
    #
    #     context = st.text_area("Enter the Context Here")
    #
    #     question = st.text_area("Enter your Question Here")
    #
    #     if context.strip() != "" and question.strip() != "":
    #         result = question_answering(question=question, context=context)
    #         s1 = json.dumps(result)
    #         d2 = json.loads(s1)
    #         generated_text = d2['answer']
    #         generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
    #         # st.write(f" Here's your Answer: {generated_text}")
    #         st.write(f"Here's your Answer:")
    #         st.markdown(f"<p style='color: #008080;'>{generated_text}</p>", unsafe_allow_html=True)

    # elif choice == "Text Completion":
    #     st.subheader("Text Completion")
    #     st.write(" Enter the uncomplete Text to complete it automatically using AI !")
    #     text_generation = pipeline("text-generation")
    #     message = st.text_area("Your Text", "Enter the Text to complete")
    #
    #     if message != "Enter the Text to complete":
    #         generator = text_generation(message)
    #         s1 = json.dumps(generator[0])
    #         d2 = json.loads(s1)
    #         generated_text = d2['generated_text']
    #         generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
    #         st.write(f" Here's your Generate Text :\n   {generated_text}")


if __name__ == '__main__':
    main()

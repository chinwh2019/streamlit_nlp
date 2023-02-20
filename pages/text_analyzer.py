import streamlit as st
import streamlit.components.v1 as stc
from spacy_streamlit import visualize_parser, visualize_ner
# Text Cleaning Pkgs
import neattext as nt
import neattext.functions as nfx
from collections import Counter
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

# Text Viz Pkgs
from wordcloud import WordCloud
from textblob import TextBlob

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import altair as alt

# External Utils
from app_utils import *

HTML_BANNER = """
    <div style="background-color:#3872fb;padding:10px;border-radius:10px;border-style:ridge;">
    <h1 style="color:white;text-align:center;">Text Analysis NLP App </h1>
    </div>
    """


@st.experimental_singleton
def load_semantic_model():
    model_sem = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    return model_sem


@st.experimental_singleton
def load_xmodel():
    model_x = CrossEncoder('cross-encoder/stsb-distilroberta-base')
    return model_x


@st.experimental_singleton
def load_model():
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return model


def get_most_common_tokens(docx, num=10):
    word_freq = Counter(docx.split())
    most_common_tokens = word_freq.most_common(num)
    return dict(most_common_tokens)


def plot_most_common_tokens(docx, num=10):
    word_freq = Counter(docx.split())
    most_common_tokens = word_freq.most_common(num)
    x, y = zip(*most_common_tokens)
    fig = plt.figure(figsize=(20, 10))
    plt.bar(x, y)
    plt.title("Most Common Tokens")
    plt.xticks(rotation=45)
    plt.show()
    st.pyplot(fig)


def plot_wordcloud(docx):
    mywordcloud = WordCloud().generate(docx)
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(mywordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)


def plot_mendelhall_curve(docx):
    word_length = [len(token) for token in docx.split()]
    word_length_count = Counter(word_length)
    sorted_word_length_count = sorted(dict(word_length_count).items())
    x, y = zip(*sorted_word_length_count)
    fig = plt.figure(figsize=(20, 10))
    plt.plot(x, y)
    plt.title("Plot of Word Length Distribution")
    plt.show()
    st.pyplot(fig)


def plot_mendelhall_curve_2(docx):
    word_length = [len(token) for token in docx.split()]
    word_length_count = Counter(word_length)
    sorted_word_length_count = sorted(dict(word_length_count).items())
    x, y = zip(*sorted_word_length_count)
    mendelhall_df = pd.DataFrame({'tokens': x, 'counts': y})
    st.line_chart(mendelhall_df['counts'])


# Functions
def generate_tags_with_spacy(docx):
    docx_with_spacy = nlp(docx)
    tagged_docx = [[[(token.text, token.pos_) for token in sent] for sent in docx_with_spacy.sents]]
    return tagged_docx


def generate_tags(docx):
    tagged_tokens = TextBlob(docx).tags
    return tagged_tokens


def generate_tags_with_textblob(docx):
    tagged_tokens = TextBlob(docx).tags
    tagged_df = pd.DataFrame(tagged_tokens, columns=['token', 'tags'])
    return tagged_df


# def plot_pos_tags(tagged_docx):
#     # Create Visualizaer, Fit ,Score ,Show
#     pos_visualizer = PosTagVisualizer()
#     pos_visualizer.fit(tagged_docx)
#     pos_visualizer.show()
#     st.pyplot()


TAGS = {
    'NN': 'green',
    'NNS': 'green',
    'NNP': 'green',
    'NNPS': 'green',
    'VB': 'blue',
    'VBD': 'blue',
    'VBG': 'blue',
    'VBN': 'blue',
    'VBP': 'blue',
    'VBZ': 'blue',
    'JJ': 'red',
    'JJR': 'red',
    'JJS': 'red',
    'RB': 'cyan',
    'RBR': 'cyan',
    'RBS': 'cyan',
    'IN': 'darkwhite',
    'POS': 'darkyellow',
    'PRP$': 'magenta',
    'PRP$': 'magenta',
    'DET': 'black',
    'CC': 'black',
    'CD': 'black',
    'WDT': 'black',
    'WP': 'black',
    'WP$': 'black',
    'WRB': 'black',
    'EX': 'yellow',
    'FW': 'yellow',
    'LS': 'yellow',
    'MD': 'yellow',
    'PDT': 'yellow',
    'RP': 'yellow',
    'SYM': 'yellow',
    'TO': 'yellow',
    'None': 'off'
}


def mytag_visualizer(tagged_docx):
    colored_text = []
    for i in tagged_docx:
        if i[1] in TAGS.keys():
            token = i[0]
            print(token)
            color_for_tag = TAGS.get(i[1])
            result = '<span style="color:{}">{}</span>'.format(color_for_tag, token)
            colored_text.append(result)
    result = ' '.join(colored_text)
    print(result)
    return result


def main():
    """Author Attribution and Verifying App"""
    stc.html(HTML_BANNER)
    menu = ["Home", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Text Analysis")
        textinput = st.text_area('Your input write here')
        st.write('Entities')
        entity_result = render_entities(textinput)
        stc.html(entity_result, height=1000, scrolling=True)

        with st.form("nlp"):
            raw_text_1 = st.text_area('Enter 1st Text Here')
            raw_text_2 = st.text_area('Enter 2nd Text Here')
            raw_text = raw_text_1 + raw_text_2
            submitted = st.form_submit_button("Submit")

            if submitted:
                if len(raw_text) > 2:
                    with st.expander("Text Analysis"):
                        token_result_df = text_analyzer(raw_text)
                        st.dataframe(token_result_df)

                    with st.expander("Entities"):
                        entity_result = render_entities(raw_text)
                        stc.html(entity_result, height=1000, scrolling=True)

                    with st.expander("POS"):
                        doc = nlp(raw_text)
                        visualize_parser(doc)

                    doc = nlp(raw_text)
                    visualize_ner(doc, labels=nlp.get_pipe("ner").labels)

                    col1, col2 = st.columns(2)
                    process_text = nfx.remove_stopwords(raw_text)

                    with col1:
                        with st.expander('Original Text'):
                            st.write(raw_text)

                        with st.expander("Preview Tagged Text"):
                            tagged_docx = generate_tags(raw_text)
                            print(tagged_docx)
                            processed_tag_docx = mytag_visualizer(tagged_docx)
                            stc.html(processed_tag_docx, scrolling=True)

                        with st.expander("Plot Word Freq"):
                            st.info("Plot For Most Common Tokens")
                            most_common_tokens = get_most_common_tokens(process_text, 20)
                            # st.write(most_common_tokens)
                            tk_df = pd.DataFrame({'tokens': most_common_tokens.keys(), 'counts': most_common_tokens.values()})
                            # tk_df = pd.DataFrame(most_common_tokens.items(),columns=['tokens','counts'])
                            # st.dataframe(tk_df)
                            # st.bar_chart(tk_df)
                            brush = alt.selection(type='interval', encodings=['x'])
                            c = alt.Chart(tk_df).mark_bar().encode(
                                x='tokens',
                                y='counts',
                                opacity=alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7)),
                            ).add_selection(brush)

                            st.altair_chart(c, use_container_width=True)

                        with st.expander("Semantic Search"):
                            model_sem = load_semantic_model()
                            query_embedding = model_sem.encode(raw_text_1)
                            passage_embedding = model_sem.encode([raw_text_2, "This is a dummy sentence"])
                            st.write(f'Similarity: {util.dot_score(query_embedding, passage_embedding)}')

                    with col2:
                        with st.expander('Processed Text'):
                            st.write(process_text)

                        with st.expander("Plot Wordcloud"):
                            st.info("word Cloud")
                            plot_wordcloud(process_text)

                        with st.expander("Plot Mendenhall Curve"):
                            plot_mendelhall_curve_2(raw_text)

                        with st.expander("Sentence Similarity"):
                            model = load_model()
                            emb1 = model.encode(raw_text_1)
                            emb2 = model.encode(raw_text_2)
                            cos_sim = util.cos_sim(emb1, emb2)
                            st.write(f"Cosine-Similarity: {cos_sim}")

                        with st.expander("Cross Similarity"):
                            xmodel = load_xmodel()
                            # emb1 = model.encode(raw_text_1)
                            # emb2 = model.encode(raw_text_2)
                            similarity_scores = xmodel.predict([raw_text_1, raw_text_2])
                            st.write(f"Cross-Similarity: {similarity_scores}")

                elif len(raw_text) == 1:
                    st.warning("Insufficient Text, Minimum must be more than 1")

    elif choice == "About":
        st.subheader("Text Analysis NLP App")


if __name__ == '__main__':
    main()
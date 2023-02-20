# Load NLP Pkgs
import spacy
import streamlit as st
from spacy import displacy
from textblob import TextBlob
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show

nlp = spacy.load("ja_core_news_trf")
# nlp = spacy.load("en_core_web_trf")


def save_html(html_content, file_name):
    with open(file_name, "w") as f:
        f.write(html_content)


def save_html2(html_list, file_name):
    # Join the HTML list into a single HTML string
    html_content = "\n".join(html_list)
    # Save the HTML file to the local drive with the specified filename
    with open(file_name, "w") as f:
        f.write(html_content)


@st.experimental_singleton
def load_bertviz_model(model_name):
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
    loaded_model = BertModel.from_pretrained(model_name, output_attentions=True)
    return loaded_tokenizer, loaded_model


@st.experimental_singleton
def load_autohf_model(model_name, attention=False):
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
    loaded_model = AutoModel.from_pretrained(model_name, output_attentions=attention)
    return loaded_tokenizer, loaded_model


@st.experimental_singleton
def load_hf_model(model_name, attention=False):
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
    loaded_model = AutoModelForMaskedLM.from_pretrained(model_name, output_attentions=attention)
    return loaded_tokenizer, loaded_model


@st.experimental_singleton
def load_spacy_en_model():
    en_model = spacy.load("en_core_web_trf")
    return en_model


@st.experimental_singleton
def load_spacy_ja_model():
    ja_model = spacy.load("ja_core_news_trf")
    return ja_model


def text_analyzer2(my_text, nlp_model):
    docx = nlp_model(my_text)
    allData = [(token.text, token.shape_, token.pos_, token.tag_, token.lemma_, token.is_alpha, token.is_stop) for token
               in docx]
    df = pd.DataFrame(allData, columns=['Token', 'Shape', 'PoS', 'Tag', 'Lemma', 'IsAlpha', 'Is_Stopword'])
    return df


def text_analyzer(my_text):
    docx = nlp(my_text)
    allData = [(token.text, token.shape_, token.pos_, token.tag_, token.lemma_, token.is_alpha, token.is_stop) for token
               in docx]
    df = pd.DataFrame(allData, columns=['Token', 'Shape', 'PoS', 'Tag', 'Lemma', 'IsAlpha', 'Is_Stopword'])
    return df


def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


# @st.cache
def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result


# Fxn to get most common tokens
def get_most_common_tokens(my_text, num=5):
    word_tokens = Counter(my_text.split())
    most_common_tokens = dict(word_tokens.most_common(num))
    return most_common_tokens


# Fxn to Get Sentiment
def get_sentiment(my_text):
    blob = TextBlob(my_text)
    sentiment = blob.sentiment
    return sentiment


# Fxn to Read PDF
from PyPDF2 import PdfFileReader
import pdfplumber


def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text


def read_pdf2(file):
    with pdfplumber.open(file) as pdf:
        page = pdf.pages[0]
        return page.extract_text()


def extract_df_data(file, filetype, percent=0.2):
    global df
    if filetype == 'tsv':
        df = pd.read_table(file)
    elif filetype == 'csv':
        df = pd.read_csv(file)
    n = int(len(df) * percent)
    extracted_data = df.sample(n=n, replace=False)
    df = df[~df.isin(extracted_data)].dropna()
    return extracted_data, df



import streamlit
import streamlit.components.v1 as stc
from spacy_streamlit import visualize_parser, visualize_ner, visualize_spans
import spacy
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher, DependencyMatcher
import stransformer_semantic_search
# External Utils
from app_utils import *


HTML_BANNER = """
    <div style="background-color:#3872fb;padding:10px;border-radius:10px;border-style:ridge;">
    <h1 style="color:white;text-align:center;">Rule-based Matching & Analysis NLP App </h1>
    </div>
    """


@Language.component("extract_person_orgs")
def extract_person_orgs(doc):
    person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    for ent in person_entities:
        head = ent.root.head
        if head.lemma_ == "work":
            preps = [token for token in head.children if token.dep_ == "prep"]
            for prep in preps:
                orgs = [t for t in prep.children if t.ent_type_ == "ORG"]
                aux = [token for token in head.children if token.dep_ == "aux"]
                past_aux = any(t.tag_ == "VBD" for t in aux)
                past = head.tag_ == "VBD" or head.tag_ == "VBG" and past_aux
                print({'person': ent, 'orgs': orgs, 'past': past})
    return doc


def main():
    stc.html(HTML_BANNER)
    menu = ["Home", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        # Store the initial value of widgets in session sate
        if "visibility" not in st.session_state:
            st.session_state.visibility = 'visible'
            st.session_state.disabled = False
            st.session_state.horizontal = False

        model_option = st.sidebar.selectbox(
            "Select a language model for analysis.",
            ("english", "japanese"),
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
        )
        model_selection = {"english": load_spacy_en_model(),
                           "japanese": load_spacy_ja_model()}
        nlp_model = model_selection.get(model_option)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.checkbox("Disable radio widget", key="disabled")
            st.checkbox("Orient radio options horizontally", key="horizontal")

        with col2:
            st.radio("Set model selection box visibility",
                     key="visibility",
                     options=["visible", "hidden", "collapsed"],
                     horizontal=st.session_state.horizontal)

        raw_text = st.text_area('Input sentence here')
        if len(raw_text) > 2:
            doc = nlp_model(raw_text)
            visualize_ner(doc, labels=nlp_model.get_pipe("ner").labels, title="Analysis")

            with st.expander("Text Analysis"):
                token_result_df = text_analyzer2(raw_text, nlp_model)
                st.dataframe(token_result_df)

            with st.expander("POS"):
                visualize_parser(doc)

            with st.expander("Token Matcher: Extract Phone Number"):
                matcher = Matcher(nlp_model.vocab)
                pattern1 = [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"},
                           {"ORTH": "-", "OP": "?"}, {"SHAPE": "ddd"}]
                matcher.add("PHONE_NUMBER", [pattern1])
                matches = matcher(doc)
                st.write(f"Extracted phone number: ")
                for match_id, start, end in matches:
                    span = doc[start:end]
                    st.write(span.text)

            with st.expander("Dependency Matcher: Extract company founder"):
                matcher = DependencyMatcher(nlp_model.vocab)

                pattern2 = [
                    {
                        "RIGHT_ID": "anchor_founded",
                        "RIGHT_ATTRS": {"ORTH": "founded"}
                    },
                    {
                        "LEFT_ID": "anchor_founded",
                        "REL_OP": ">",
                        "RIGHT_ID": "founded_subject",
                        "RIGHT_ATTRS": {"DEP": "nsubj"},
                    },
                    {
                        "LEFT_ID": "anchor_founded",
                        "REL_OP": ">",
                        "RIGHT_ID": "founded_object",
                        "RIGHT_ATTRS": {"DEP": "dobj"},
                    },
                    {
                        "LEFT_ID": "founded_object",
                        "REL_OP": ">",
                        "RIGHT_ID": "founded_object_modifier",
                        "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}},
                    }
                ]

                matcher.add("FOUNDED", [pattern2])
                matches = matcher(doc)
                match_id, token_ids = matches[0]
                for i in range(len(token_ids)):
                    st.write(pattern2[i]["RIGHT_ID"] + ":", doc[token_ids[i]].text)

            with st.expander("Phrase Matcher: Extract phrases from text"):
                matcher = PhraseMatcher(nlp_model.vocab)
                terms = st.multiselect('Select key phrase for extraction.',
                                       ["Barack Obama", "Angela Merkel", "Washington, D.C.", "Kishida", "Jayden Chin"])
                patterns = [nlp_model.make_doc(text) for text in terms]
                matcher.add("TerminologyList", patterns)
                matches = matcher(doc)
                span_all = []
                for match_id, start, end in matches:
                    span = doc[start:end]
                    span_all.append(span)
                    span.label_ = "Extracted_Info"
                doc.spans["sc"] = span_all
                visualize_spans(doc, spans_key="sc", displacy_options={"colors": {"Extracted_Info": "#09a3d5"}})

            # with st.expander("Entities, POS, DP extraction: Extract pass tense info"):
            #     # To make the entities easier to work with, we'll merge them into single tokens
            #     nlp_model.add_pipe("merge_entities")
            #     nlp_model.add_pipe("extract_person_orgs")

        elif len(raw_text) == 1:
            st.warning("Insufficient text, minimum text must be more than 1")

    elif choice == "About":
        st.subheader("Rule-based linguistic analysis & applications")


if __name__ == '__main__':
    main()

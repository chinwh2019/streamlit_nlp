import streamlit.components.v1 as components
from app_utils import *
from bertviz import head_view, model_view

def main():
    st.title("Transformer visualizer")
    text = st.text_area(value="time flies like an arrow", label="Input text")
    text2 = st.text_area(value="time flies like an arrow", label="Input another text")
    model_type = st.sidebar.radio(
        "Select type of transformer",
        ('encoder', 'encoder_decoder', 'vizbert')
    )
    if model_type == "encoder":
        model_option = st.sidebar.selectbox(
            "Select a transformer model for visualization.",
            ("deberta-large", "deberta-large-wwm", "bert-ja", "bert-uncased")
        )
        model_selection = {
            "deberta-large": 'ku-nlp/deberta-v2-large-japanese',
            "deberta-large-wwm": 'ku-nlp/roberta-large-japanese-char-wwm',
            "bert-ja": "cl-tohoku/bert-base-japanese",
            "bert-uncased": "bert-base-uncased"
        }
        nlp_tokenizer, nlp_model = load_hf_model(model_selection.get(model_option), attention=True)
        inputs = nlp_tokenizer.encode(text, return_tensors='pt')
        outputs = nlp_model(inputs)
        attention = outputs[-1]  # Output includes attention weights when output_attentions=True
        tokens = nlp_tokenizer.convert_ids_to_tokens(inputs[0])
        html_head_view = head_view(attention, tokens, html_action="return")
        with st.expander("Head view"):
            components.html(html_head_view.data, height=1000)
        html_model_view = model_view(attention, tokens, html_action="return")
        with st.expander("Model view"):
            components.html(html_model_view.data, height=3000)

    elif model_type == "vizbert":
        model_option = st.sidebar.selectbox(
            "Select a transformer model for visualization.",
            ("bert-uncased",)
        )
        model_selection = {
            "bert-uncased": "bert-base-uncased",
        }
        nlp_tokenizer, nlp_model = load_bertviz_model(model_selection.get(model_option))
        html_neuron_view = show(nlp_model, "bert", nlp_tokenizer, text, display_mode="light", layer=0, head=8,
                                html_action="return")
        components.html(html_neuron_view.data, height=1500)


if __name__ == "__main__":
    main()

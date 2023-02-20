import torch
import MeCab
from app_utils import *
from transformers import pipeline
import subprocess
from unidecode import unidecode
from pyknp import Juman


def main():
    st.title("deberta-v2 (ku-lab) - Fill Mask")

    model_option = st.sidebar.selectbox(
        "Select a language model for analysis.",
        ("deberta-large", "deberta-large-wwm"),
    )
    model_selection = {"deberta-large": 'ku-nlp/deberta-v2-large-japanese',
                       "deberta-large-wwm": 'ku-nlp/roberta-large-japanese-char-wwm'}

    nlp_model = load_hf_model(model_selection.get(model_option))
    tokenizer, model = nlp_model
    text = st.text_area("Enter Text Here", value='京都大学で自然言語処理を[MASK]する')

    with st.expander("Result"):
        encoding = tokenizer(text, return_tensors='pt')
        output = model(**encoding).logits
        mask_token_index = torch.where(encoding["input_ids"] == tokenizer.mask_token_id)[1]
        mask_token_logits = output[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        for token in top_5_tokens:
            correct_sentence = text.replace(tokenizer.mask_token, tokenizer.decode([token]))
            m = MeCab.Tagger("-Owakati")
            words = m.parse(correct_sentence).strip().split()
            highlight_word = tokenizer.decode([token])
            marked_up_words = []
            for word in words:
                if word == highlight_word:
                    marked_up_word = f"<span style='color:red'>{word}</span>"
                else:
                    marked_up_word = word
                marked_up_words.append(marked_up_word)
            st.markdown("".join(marked_up_words), unsafe_allow_html=True)
            #st.markdown(f"<span style='color:red'>{correct_sentence}</span>", unsafe_allow_html=True)

    with st.expander("Result-HuggingFace-pipeline"):
        mask_filler = pipeline(
            "fill-mask", model=model_selection.get(model_option)
        )
        preds = mask_filler(text)
        for pred in preds:
            st.write(f">>> {pred['sequence']}")

if __name__ == "__main__":
    main()

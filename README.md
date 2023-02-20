# Transformer Visualizer

This is a Streamlit app that allows users to visualize the attention of different transformer models. Users can input one or more texts and select the type and name of the transformer model to visualize.

## Installation

1. Clone this repository: `git clone https://github.com/<USERNAME>/<REPO_NAME>.git`
2. Navigate to the repository: `cd <REPO_NAME>`
3. Install the required packages: `pip install -r requirements.txt`

## Usage

1. Run the Streamlit app: `streamlit run main.py`
2. In the app, select the type of transformer from the sidebar radio button.
3. Select the name of the transformer model for visualization from the dropdown.
4. Input one or two sentences to visualize the attention of the model.

## Supported Models

The app currently supports the following transformer models:

- Encoder: BERT, DeBERTa-Large-JA, DeBERTa-Large-WWM-JA, BERT-JA
- Encoder-Decoder: BART, T5
- VizBERT: BERT-Base-Uncased

## Attribution

This app was created using the following libraries:

- [Streamlit](https://streamlit.io/)
- [BERTViz](https://github.com/jessevig/bertviz)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## License

TBD

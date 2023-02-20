# !/usr/bin/env python3
# coding: utf-8

"""
This code is a streamlit app that allows the user to visualize the attention of different transformer models.
The user can input one or more texts and select the type and name of the transformer model to visualize.
The app uses the utils module to load the models and generate the html for the visualization.
The app also uses the components module to display the html in the streamlit interface.
"""

# Import the necessary modules
import streamlit.components.v1 as components
from app_utils import *
from bertviz import head_view, model_view


# Define the main function of the app
def main():
    # Display the title of the app
    st.title("Transformer visualizer")
    # Create a sidebar radio button for the user to select the type of transformer
    model_type = st.sidebar.radio(
        "Select type of transformer",
        ('encoder', 'encoder_decoder', 'vizbert')
    )
    # Use a dictionary to map the model type to the corresponding visualization function
    visualization_functions = {
        "encoder": visualize_encoder,
        "encoder_decoder": visualize_encoder_decoder,
        "vizbert": visualize_vizbert
    }
    # Use the get method to execute the visualization function based on the model type
    visualization_functions.get(model_type, lambda x: None)()


# Define the function to visualize the encoder model
def visualize_encoder():
    # Use a dictionary to map the model option to the corresponding model name
    model_selection = {
        "bert-uncased": "bert-base-uncased",
        "deberta-large-ja": 'ku-nlp/deberta-v2-large-japanese',
        "deberta-large-wwm-ja": 'ku-nlp/roberta-large-japanese-char-wwm',
        "bert-ja": "cl-tohoku/bert-base-japanese",
    }
    # Use a list to store the model options
    model_options = list(model_selection.keys())
    # Use a selectbox to choose the model option
    model_option = st.sidebar.selectbox(
        "Select a transformer model for visualization.",
        model_options
    )
    # Use the model selection dictionary to get the model name
    model_name = model_selection.get(model_option)
    # Load the model and tokenizer with attention
    nlp_tokenizer, nlp_model = load_hf_model(model_name, attention=True)
    # Create a text area for the user to input the first sentence
    text_1 = st.text_area(value="time flies like an arrow", label="Input the first sentence")
    # Create a text area for the user to input the second sentence
    text_2 = st.text_area(value="fruit flies like a banana", label="Input the second sentence")
    # Visualize the attention
    visualize_attention(text_1, text_2, nlp_tokenizer, nlp_model)


# Define the function to visualize the encoder-decoder model
def visualize_encoder_decoder():
    model_selection = {
        "bart": 'facebook/bart-large-mnli',
        "T5": 'Helsinki-NLP/opus-mt-en-de',
    }
    # Use a list to store the model options
    model_options = list(model_selection.keys())
    # Use a selectbox to choose the model option
    model_option = st.sidebar.selectbox(
        "Select a transformer model for visualization.",
        model_options
    )
    # Use the model selection dictionary to get the model name
    model_name = model_selection.get(model_option)
    # Initialize tokenizer and model. Be sure to set output_attentions=True.
    tokenizer, model = load_autohf_model(model_name, attention=True)
    # Create a text area for the user to input the first sentence
    text_1 = st.text_area(value="She sees the small elephant.", label="Input the first sentence")
    # Create a text area for the user to input the second sentence
    text_2 = st.text_area(value="Sie sieht den kleinen Elefanten.", label="Input the second sentence")
    visualize_attention_encoder_encoder(text_1, text_2, tokenizer, model)


# Define the function to visualize the vizbert model
def visualize_vizbert():
    # Use a constant to store the model name
    text_1 = st.text_area(value="time flies like an arrow", label="Input a sentence")
    model_name = "bert-base-uncased"
    # Load the model and tokenizer
    nlp_tokenizer, nlp_model = load_bertviz_model(model_name)

    with st.expander("Neuron view"):
        html_neuron_view = show(nlp_model, "bert", nlp_tokenizer, text_1, display_mode="light", layer=0, head=8,
                                html_action="return")
        components.html(html_neuron_view.data, height=800, scrolling=True)

    # Create a sidebar to save the visualization in html format
    st.sidebar.markdown("## Save the visualization in html format")
    # Create a text input for the user to specify the filename
    file_name = st.sidebar.text_input("Enter the filename (include .html extension)")
    # Create a button to save the visualization in html format
    if st.sidebar.button("Save HTML"):
        # Save the visualization in html format
        html_content = html_neuron_view.data
        save_html(html_content, file_name)
        st.sidebar.download_button(label="Download HTML", data=html_content, file_name=file_name, mime="text/html")


# Define the function to visualize the attention encoder decoder
def visualize_attention_encoder_encoder(text_1, text_2, tokenizer, model):
    # get encoded input vectors
    encoder_input_ids = tokenizer(text_1, return_tensors="pt",
                                  add_special_tokens=True).input_ids
    # create ids of encoded input vectors
    with tokenizer.as_target_tokenizer():
        decoder_input_ids = tokenizer(text_2, return_tensors="pt",
                                      add_special_tokens=True).input_ids
    outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
    encoder_text = tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
    decoder_text = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])

    with st.expander("Head view"):
        html_head_view = head_view(
            encoder_attention=outputs.encoder_attentions,
            decoder_attention=outputs.decoder_attentions,
            cross_attention=outputs.cross_attentions,
            encoder_tokens=encoder_text,
            decoder_tokens=decoder_text,
            html_action="return"
        )
        # Display the html in the expander
        components.html(html_head_view.data, height=800, scrolling=True)

    with st.expander("Model view"):
        html_model_view = model_view(
            encoder_attention=outputs.encoder_attentions,
            decoder_attention=outputs.decoder_attentions,
            cross_attention=outputs.cross_attentions,
            encoder_tokens=encoder_text,
            decoder_tokens=decoder_text,
            html_action="return"
        )
        # Display the html in the expander
        components.html(html_model_view.data, height=800, scrolling=True)

    # Create a sidebar to save the visualization in html format
    st.sidebar.markdown("## Save the visualization in html format")
    # Create a text input for the user to specify the filename
    file_name = st.sidebar.text_input("Enter the filename (include .html extension)")
    # Create a button to save the visualization in html format
    if st.sidebar.button("Save HTML"):
        # Save the visualization in html format
        html_content = [html_head_view.data, html_model_view.data]
        save_html2(html_content, file_name)
        html_content = html_head_view.data + html_model_view.data
        st.sidebar.download_button(label="Download HTML", data=html_content, file_name=file_name, mime="text/html")


# Define the function to visualize the attention encoder
def visualize_attention(text_1, text_2, nlp_tokenizer, nlp_model):
    # Encode the texts
    inputs = nlp_tokenizer.encode_plus(text_1, text_2, return_tensors='pt')
    # Get the outputs and attention
    outputs = nlp_model(**inputs)
    attention = outputs[-1]  # Output includes attention weights when output_attentions=True
    # Convert the inputs to tokens
    tokens = nlp_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # Generate the html for head view and model view
    html_head_view = head_view(attention, tokens, html_action="return")
    html_model_view = model_view(attention, tokens, html_action="return")

    # Create an expander for the head view
    with st.expander("Head view"):
        # Display the html in the expander
        components.html(html_head_view.data, height=800, width=800, scrolling=True)

    # Create an expander for the model view
    with st.expander("Model view"):
        # Display the html in the expander
        components.html(html_model_view.data, height=800, scrolling=True)

    # Create a sidebar to save the visualization in html format
    st.sidebar.markdown("## Save the visualization in html format")
    # Create a text input for the user to specify the filename
    file_name = st.sidebar.text_input("Enter the filename (include .html extension)")
    # Create a button to save the visualization in html format
    if st.sidebar.button("Save HTML"):
        # Save the visualization in html format
        html_content = [html_head_view.data, html_model_view.data]
        save_html2(html_content, file_name)
        html_content = html_head_view.data + html_model_view.data
        st.sidebar.download_button(label="Download HTML", data=html_content, file_name=file_name, mime="text/html")


# Run the main function
if __name__ == "__main__":
    main()

import streamlit as st
import whisper

model = whisper.load_model("base")

# Define the main function
def main():
    st.title("Speech-to-Text Transcription")

    # Get the audio file from the user
    audio_file = st.file_uploader("Select an audio file (mp3 or wav)", type=["mp3", "wav"])
    if audio_file is None:
        st.write("No file selected.")
        return

    # Transcribe the audio file to text
    audio_file.seek(0)
    data = audio_file.read()
    result = model.transcribe(data)

    text = result["data"]
    st.write("Transcription: ")
    st.write(text)


# Run the Streamlit app
if __name__ == "__main__":
    main()

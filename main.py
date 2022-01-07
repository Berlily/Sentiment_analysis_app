import pyaudio
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
import streamlit as st
import plot_data
import os

from utils import extract_feature

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16  # Sound is stored in binary, as is everything related to computers. In order to know where an integer starts and ends, there are different methods used. PyAudio uses a fixed size of bits.
# paInt16 is basically a signed 16-bit binary string.
RATE = 16000

SILENCE = 30


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')  # 'h' means signed shorts in C (and int in Python) with minimum size in bytes:2
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def save_audio(audio_file):
    if not os.path.exists("audio"):
        os.makedirs("audio")
    path = os.path.join("audio", "tmp.wav")

    with open(path, "wb") as outfile:
        bytes_data = audio_file.read()
        outfile.write(bytes_data)
    return path


if __name__ == "__main__":
    # load the saved model (after training)
    model = pickle.load(open("result/mlp_classifier.model", "rb"))

    # NOW the interface!
    # 1. Title
    st.title("Sentiment Analysis App")

    st.write('you\'ll find the menu in the sidebar:)')
    st.write('')
    st.write('')
    st.write('')

    # 2. Layout.
    # Sidebar
    st.sidebar.caption(
        "A simple deep learning app to predict the sentiment of the text or the speech"
    )
    st.sidebar.write('')
    add_selectbox = st.sidebar.selectbox(
        "You want to analyse...",
        ("Audio", "Text"))

    if add_selectbox == "Text":
        # Declare a form to analyse text
        form = st.form(key="my_form")
        review = form.text_input(label="Enter your text here")
        submit = form.form_submit_button(label="Make Prediction")

        if submit:
            # make prediction from the input text
            # Display results
            st.header("Result")
            plot_data.plot_text_pie_chart(review)

    if add_selectbox == "Audio":  # if a user chooses audio in the selectbox

        with st.container():
            col1, col2 = st.columns(2)
            # audio_file = None
            # path = None
            with col1:
                # TO CHECK!!!!!!!!! I MEAN, CHECK THE EXTENSION TYPE in the line below. WAV ONLY? OR MP3 & OGG(????) ALSO FINE ?
                audio_file = st.file_uploader("Upload audio file", type=['wav'])
                if audio_file is not None:  # if user chose to upload a file from their computer
                    path = save_audio(audio_file)

                    features = extract_feature(path, mfcc=True, chroma=True, mel=True).reshape(1, -1)

                    result = model.predict(features)[0]
                    # show the result !
                    st.header("Result")
                    st.write(result)

                else:  # if user chose to record their own voice
                    st.write('')
                    if st.button("Test my own voice!"):
                        filename = "test.wav"
                        # record the file (start talking)
                        record_to_file(filename)
                        # extract features and reshape it
                        features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
                        # predict
                        result = model.predict(features)[0]
                        # show the result !
                        st.header("Result")
                        st.write(result)







            # with col2:
            #     if audio_file is not None:
            #         fig = plt.figure(figsize=(10, 2))
            #         fig.set_facecolor('#d1d1e0')
            #         plt.title("Wave-form")
            #         librosa.display.waveplot(wav, sr=44100)
            #         plt.gca().axes.get_yaxis().set_visible(False)
            #         plt.gca().axes.get_xaxis().set_visible(False)
            #         plt.gca().axes.spines["right"].set_visible(False)
            #         plt.gca().axes.spines["left"].set_visible(False)
            #         plt.gca().axes.spines["top"].set_visible(False)
            #         plt.gca().axes.spines["bottom"].set_visible(False)
            #         plt.gca().axes.set_facecolor('#d1d1e0')
            #         st.write(fig)
            #     else:
            #         pass

# 2. show sample wave
# st.write(plot_data.show_wave("test.wav"))

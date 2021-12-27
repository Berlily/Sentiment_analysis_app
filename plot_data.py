import matplotlib as mpl
import matplotlib.pyplot as plt

import librosa.display
import numpy as np
import pandas as pd
import text2emotion as te
import streamlit as st


def show_wave(data):
    plt.figure(figsize=(10, 3))
    librosa.display.waveplot(data)
    plt.show()


def plot_text_pie_chart(text):
    # first task is to work with our emotion dictionary
    emotion_dict = te.get_emotion(text)
    # I've muted the following lines in order to fix this: "ValueError: cannot convert float NaN to integer"
    # for i in emotion_dict:
    #     emotion_dict[i] *= 100
    #     emotion_dict[i] = int(emotion_dict[i])

    # second task is to plot this dictionary in pie chart
    y = np.array(list(emotion_dict.values()))
    sizes = list(emotion_dict.values())
    my_labels = list(emotion_dict.keys())

    fig, y = plt.subplots()
    y.pie(sizes, labels=my_labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
    y.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(loc="upper left")

    st.pyplot(fig)



if __name__ == "__main__":
    plot_text_pie_chart('Do you consider the forms of introduction, and the stress that is laid on them, as nonsense? I cannot quite agree with you there. What say you, Mary? for you are a young lady of deep reflection, I know, and read great books and make extracts.')




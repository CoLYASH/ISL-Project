import os
import cv2
import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import timeit  # Import timeit for performance measurement
from datetime import datetime  # For timestamping performance metrics

# Download required NLTK data with error handling
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    st.warning(f"Couldn't download NLTK data: {e}")

# Dataset paths
base_path = r"C:\Users\Yash Dhoney\PycharmProjects\pythonProject\ISL_CSLRT_Corpus new\ISL_CSLRT_Corpus"
sentence_videos_path = os.path.join(base_path, "Videos_Sentence_Level")
word_videos_path = os.path.join(base_path, "new_ones", "Words")
letter_videos_path = os.path.join(base_path, "new_ones", "Letters")
number_videos_path = os.path.join(base_path, "new_ones", "Numbers")
frames_word_level_path = os.path.join(base_path, "Frames_Word_Level")

st.title("🖐 Indian Sign Language Translator")
st.write("Convert speech to ISL animations")


# ====================== CORE FUNCTIONS ======================
def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Speak in Hindi or Marathi...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio, language="hi")
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio.")
    except sr.RequestError:
        st.error("❌ Could not connect to Google API.")
    return None


def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        st.error(f"❌ Translation Error: {e}")
        return None


def get_video_or_frames_for_text(input_text):
    """Finds the best matching ISL video or frames for the input text."""
    video_mapping = {}
    input_text_cleaned = input_text.lower().strip()

    # 1️⃣ Check for sentence-level video
    for folder in os.listdir(sentence_videos_path):
        if folder.lower().strip() == input_text_cleaned:
            folder_path = os.path.join(sentence_videos_path, folder)
            for filename in os.listdir(folder_path):
                if filename.lower().startswith(input_text_cleaned):
                    video_path = os.path.join(folder_path, filename)
                    return {input_text: [video_path]}

    # 2️⃣ Check for word-level video or frames
    words = input_text.split()
    for word in words:
        word_lower = word.lower()
        word_video = os.path.join(word_videos_path, f"{word_lower}.mp4")
        if os.path.exists(word_video):
            video_mapping[word] = [word_video]
            continue

        word_frame_dir = os.path.join(frames_word_level_path, word.upper())
        if os.path.exists(word_frame_dir):
            frame_files = sorted(
                [os.path.join(word_frame_dir, f) for f in os.listdir(word_frame_dir)],
                key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0
            )
            if frame_files:
                video_mapping[word] = frame_files

        # 3️⃣ Check for letters if no word match found
        if word not in video_mapping:
            letter_videos = {}
            for char in word:
                letter_video = os.path.join(
                    number_videos_path if char.isdigit() else letter_videos_path,
                    f"{char.upper()}.mp4"
                )
                if os.path.exists(letter_video):
                    letter_videos[char] = letter_video

            if letter_videos:
                video_mapping[word] = letter_videos

    return video_mapping if video_mapping else None


def play_videos_or_frames(video_mapping):
    """Play videos or frames sequentially."""
    stframe = st.empty()

    for word, items in video_mapping.items():
        st.subheader(f"🎥 Showing: {word}")

        if isinstance(items, list):
            if items[0].endswith(".mp4"):
                video_list = items
            else:
                for frame in items:
                    frame_image = cv2.imread(frame)
                    frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_image, channels="RGB", use_container_width=True)
                    time.sleep(1)
                continue
        else:
            video_list = items.items()

        for item in video_list:
            if isinstance(item, tuple):
                letter, video = item
                st.subheader(f"🎥 Playing: {letter}")
            else:
                video = item

            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                st.error(f"⚠️ Error: Could not open {video}")
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB", use_container_width=True)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cap.release()
                    return
            cap.release()
    stframe.empty()


# ====================== NLP VISUALIZATION FUNCTIONS ======================
def plot_word_frequency(text):
    try:
        words = word_tokenize(text.lower())
        word_counts = Counter(words)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(word_counts.keys(), word_counts.values())
        ax.set_title("Word Frequency Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Couldn't generate word frequency: {e}")


def plot_sentence_length(text):
    try:
        sentences = nltk.sent_tokenize(text)
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(sentences)), lengths, color='skyblue')
        ax.set_title("Sentence Length")
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Couldn't generate sentence length analysis: {e}")





def show_text_metrics(text):
    try:
        words = word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        metrics = {
            "Word Count": len(words),
            "Unique Words": len(set(words)),
            "Sentence Count": len(sentences),
            "Avg Word Length": f"{sum(len(word) for word in words) / len(words):.2f}",
            "Avg Sentence Length": f"{len(words) / len(sentences):.2f}"
        }
        st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=["Value"]))
    except Exception as e:
        st.warning(f"Couldn't generate text metrics: {e}")


# ====================== MODIFIED SIDEBAR CONTROLS ======================
with st.sidebar:
    st.header("NLP Visualizations")
    show_word_freq = st.checkbox("Word Frequency", False, key="word_freq")
    show_sent_len = st.checkbox("Sentence Length", False, key="sent_len")
    show_metrics = st.checkbox("Text Metrics", False, key="text_metrics")
    show_perf_metrics = st.checkbox("Performance Metrics", False, key="perf_metrics")

# ====================== MODIFIED MAIN APP LOGIC ======================
if st.button("🎤 Speak Now", key="speak_button"):
    # Initialize performance tracking
    perf_data = {
        'recognition_time': None,
        'translation_time': None,
        'matching_time': None,
        'playback_time': None
    }

    # Speech recognition with timing
    recog_start = timeit.default_timer()
    spoken_text = get_speech_input()
    perf_data['recognition_time'] = timeit.default_timer() - recog_start

    if spoken_text:
        st.success(f"🗣 Recognized: {spoken_text}")

        # NLP Analysis
        if show_word_freq or show_sent_len or show_metrics:
            with st.expander("NLP Analysis", expanded=True):
                if show_metrics:
                    show_text_metrics(spoken_text)
                if show_word_freq:
                    plot_word_frequency(spoken_text)
                if show_sent_len:
                    plot_sentence_length(spoken_text)

        # Translation with timing
        trans_start = timeit.default_timer()
        translated_text = translate_to_english(spoken_text)
        perf_data['translation_time'] = timeit.default_timer() - trans_start

        if translated_text:
            st.info(f"🌍 Translated: {translated_text}")

            # Video matching with timing
            match_start = timeit.default_timer()
            result = get_video_or_frames_for_text(translated_text)
            perf_data['matching_time'] = timeit.default_timer() - match_start

            if result:
                st.success("✅ Showing ISL Animations or Frames...")

                # Playback with timing
                play_start = timeit.default_timer()
                play_videos_or_frames(result)
                perf_data['playback_time'] = timeit.default_timer() - play_start

                # Show performance metrics if enabled
                if show_perf_metrics:
                    with st.expander("Performance Metrics", expanded=False):
                        # Speech-to-Text Accuracy
                        st.subheader("🔊 Speech Recognition")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Recognition Time", f"{perf_data['recognition_time']:.2f}s")

                        # Translation Quality
                        st.subheader("🌍 Translation")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Translation Time", f"{perf_data['translation_time']:.2f}s")

                        # ISL Matching
                        st.subheader("🖐 ISL Matching")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Matching Time", f"{perf_data['matching_time']:.2f}s")

                        # Execution Times Visualization
                        st.subheader("⏱ Execution Timeline")
                        times = {
                            "Speech Recognition": perf_data['recognition_time'],
                            "Translation": perf_data['translation_time'],
                            "ISL Matching": perf_data['matching_time']
                        }
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.bar(times.keys(), times.values(), color=['#4C72B0', '#DD8452', '#55A868'])
                        ax.set_ylabel("Seconds")
                        st.pyplot(fig)
                        plt.close()
            else:
                st.warning("❌ No ISL data found.")
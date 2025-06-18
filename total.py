import os
import cv2
import speech_recognition as sr
from deep_translator import GoogleTranslator

# Dataset paths
base_path = r"C:\Users\Yash Dhoney\PycharmProjects\pythonProject\ISL_CSLRT_Corpus new\ISL_CSLRT_Corpus"
sentence_videos_path = os.path.join(base_path, "Videos_Sentence_Level")
word_videos_path = os.path.join(base_path, "new_ones", "Words")
letter_videos_path = os.path.join(base_path, "new_ones", "Letters")
number_videos_path = os.path.join(base_path, "new_ones", "Numbers")


def get_speech_input():
    """Captures speech and returns transcribed text."""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Say something in Hindi or Marathi...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
        audio = recognizer.listen(source)

    # Save recorded audio (for debugging)
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())

    try:
        # Recognize speech using Google API (supports Hindi & Marathi)
        text = recognizer.recognize_google(audio, language="hi")  # 'hi' supports Hindi + Marathi
        print(f"🗣 Recognized Speech: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand the audio. Try again.")
        return None
    except sr.RequestError:
        print("❌ Could not connect to Google API. Check your internet.")
        return None


def translate_to_english(text):
    """Translates text from Hindi/Marathi to English."""
    try:
        translated_text = GoogleTranslator(source="auto", target="en").translate(text)
        print(f"🌍 Translated to English: {translated_text}")
        return translated_text
    except Exception as e:
        print(f"❌ Translation Error: {e}")
        return None


def get_video_for_text(input_text):
    """
    Fetches ISL animations for the given input text.
    Returns a dictionary mapping words/letters to video paths.
    """
    video_mapping = {}

    # Check if a direct sentence video exists
    sentence_video = os.path.join(sentence_videos_path, f"{input_text}.mp4")
    if os.path.exists(sentence_video):
        return {input_text: [sentence_video]}  # ✅ Direct sentence match found

    words = input_text.split()

    for word in words:
        word_video = os.path.join(word_videos_path, f"{word}.mp4")
        if os.path.exists(word_video):
            video_mapping[word] = [word_video]
        else:
            # If a word is missing, break it into letters/numbers
            letter_videos = []
            for char in word:
                if char.isdigit():
                    letter_video = os.path.join(number_videos_path, f"{char}.mp4")
                else:
                    letter_video = os.path.join(letter_videos_path, f"{char.upper()}.mp4")

                if os.path.exists(letter_video):
                    letter_videos.append(letter_video)

            if letter_videos:
                video_mapping[word] = letter_videos

    return video_mapping if video_mapping else None


def play_videos(video_mapping):
    """
    Plays videos one after another using OpenCV and displays word/letter mappings.
    """
    for word, videos in video_mapping.items():
        print(f"\n🎥 Showing ISL animation for: {word}")  # Display word/letter

        for video in videos:
            cap = cv2.VideoCapture(video)

            if not cap.isOpened():
                print(f"⚠️ Error: Could not open {video}")
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow(f"ISL Animation - {word}", frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit early
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        # Step 1: Get speech input
        spoken_text = get_speech_input()
        if not spoken_text:
            continue  # Try again if speech was not recognized

        # Step 2: Translate to English
        translated_text = translate_to_english(spoken_text)
        if not translated_text:
            continue  # Skip if translation fails

        # Step 3: Fetch ISL animations
        result = get_video_for_text(translated_text)

        if result:
            print(f"\n✅ Found {sum(len(v) for v in result.values())} ISL animations for:")
            for word, videos in result.items():
                print(f"  - {word}: {len(videos)} video(s)")

            play_videos(result)  # Step 4: Play ISL animations
        else:
            print("❌ No ISL animation found for the given input.")

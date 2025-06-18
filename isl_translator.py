import os
import cv2

# Define dataset paths
base_path = r"C:\Users\Yash Dhoney\PycharmProjects\pythonProject\ISL_CSLRT_Corpus new\ISL_CSLRT_Corpus"
sentence_videos_path = os.path.join(base_path, "Videos_Sentence_Level")
word_videos_path = os.path.join(base_path, "new_ones", "Words")
letter_videos_path = os.path.join(base_path, "new_ones", "Letters")
number_videos_path = os.path.join(base_path, "new_ones", "Numbers")


def get_video_for_text(input_text):
    """
    Fetches the most suitable ISL animation for the given input text.
    Returns a list of video paths to be displayed in sequence.
    """
    sentence_video = os.path.join(sentence_videos_path, f"{input_text}.mp4")
    if os.path.exists(sentence_video):
        return [sentence_video]  # ✅ Direct sentence match found

    words = input_text.split()
    video_paths = []

    for word in words:
        word_video = os.path.join(word_videos_path, f"{word}.mp4")
        if os.path.exists(word_video):
            video_paths.append(word_video)
        else:
            # If a word is missing, break it into letters/numbers
            for char in word:
                if char.isdigit():
                    letter_video = os.path.join(number_videos_path, f"{char}.mp4")
                else:
                    letter_video = os.path.join(letter_videos_path, f"{char.upper()}.mp4")

                if os.path.exists(letter_video):
                    video_paths.append(letter_video)

    return video_paths if video_paths else None


def play_videos(video_paths):
    """
    Plays videos one after another using OpenCV.
    """
    for video in video_paths:
        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            print(f"Error: Could not open {video}")
            continue

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("ISL Animation", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit early
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter text to translate into ISL (or type 'exit' to quit): ").strip()

        if user_input.lower() == "exit":
            print("Exiting program.")
            break

        result = get_video_for_text(user_input)

        if result:
            print(f"Playing {len(result)} ISL animations...")
            play_videos(result)
        else:
            print("No ISL animation found for the given input.")

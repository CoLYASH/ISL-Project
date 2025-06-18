import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something...")
    r.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
    speech = r.listen(source)

# Save audio for debugging
with open("recorded_audio.wav", "wb") as f:
    f.write(speech.get_wav_data())

try:
    text = r.recognize_google(speech)
    print("You said: " + text)
except sr.UnknownValueError:
    print("I couldn't understand the audio. Try speaking more clearly.")
except sr.RequestError:
    print("Could not connect to Google API. Check your internet connection.")

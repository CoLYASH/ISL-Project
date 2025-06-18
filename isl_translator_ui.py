import sys
import threading
import speech_recognition as sr
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTextEdit, QVBoxLayout
from deep_translator import GoogleTranslator

class ISLTranslatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Speech to ISL Translator")
        self.setGeometry(100, 100, 500, 400)

        layout = QVBoxLayout()

        self.speech_label = QLabel("Recognized Speech:")
        layout.addWidget(self.speech_label)
        self.speech_textbox = QTextEdit()
        layout.addWidget(self.speech_textbox)

        self.translation_label = QLabel("Translated English:")
        layout.addWidget(self.translation_label)
        self.translation_textbox = QTextEdit()
        layout.addWidget(self.translation_textbox)

        self.start_button = QPushButton("Start Speech Recognition")
        self.start_button.clicked.connect(self.start_speech_recognition)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def start_speech_recognition(self):
        self.speech_textbox.setPlainText("Listening...")
        thread = threading.Thread(target=self.recognize_speech)
        thread.start()

    def recognize_speech(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source)
                speech_text = recognizer.recognize_google(audio, language='hi')  # Hindi
                self.speech_textbox.setPlainText(speech_text)
                translated_text = GoogleTranslator(source='auto', target='en').translate(speech_text)
                self.translation_textbox.setPlainText(translated_text)
            except sr.UnknownValueError:
                self.speech_textbox.setPlainText("Could not understand the audio.")
            except sr.RequestError:
                self.speech_textbox.setPlainText("Speech recognition service error.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ISLTranslatorApp()
    window.show()
    sys.exit(app.exec())

import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

LANG_MAP = {
    "en": {"sr": "en-US", "gtts": "en"},
    "bn": {"sr": "bn-BD", "gtts": "bn"},
}


def transcribe_audio(audio_bytes: bytes, lang: str = "en") -> str:
    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            data = recognizer.record(source)
        sr_lang = LANG_MAP.get(lang, LANG_MAP["en"])["sr"]
        return recognizer.recognize_google(data, language=sr_lang)
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def text_to_speech(text: str, lang: str = "en") -> bytes:
    mp3_path = tempfile.mktemp(suffix=".mp3")
    try:
        gtts_lang = LANG_MAP.get(lang, LANG_MAP["en"])["gtts"]
        gTTS(text=text, lang=gtts_lang).save(mp3_path)
        with open(mp3_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(mp3_path)
        except Exception:
            pass

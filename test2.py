import speech_recognition as sr
import sounddevice as sd
from gtts import gTTS
from pyvi import ViTokenizer, ViPosTagger
from pydub import AudioSegment
from scipy.io import wavfile
import json
import random

with open('datachatbot.json','r', encoding="utf-8") as file:
    data = json.load(file)
def compare(user_input, data):
    max_matched_words = 0
    best_answer = None
    question_in_data=None
    tokens, tags = ViPosTagger.postagging(ViTokenizer.tokenize(user_input))
    tokens = set(tokens)

    for item in data:
        question_tokens, _ = ViPosTagger.postagging(ViTokenizer.tokenize(item['question']))
        matched_words = len(tokens.intersection(set(question_tokens)))
        question=item['question']
        try:
            if matched_words > max_matched_words:
                max_matched_words = matched_words
                best_answer = random.choice(item['answer'])
                question_in_data=question
        except:
            best_answer='Xin lỗi tôi chưa hiểu bạn nói gì.'
    return max_matched_words,best_answer,question_in_data


def play_wav(wav_file_path, speed_factor=1.0):
    # Chuyển đổi tệp MP3 sang WAV
    audio = AudioSegment.from_mp3('output.mp3')
    audio.export(wav_file_path, format="wav")
    # Đọc tệp WAV với thư viện scipy
    fs, data = wavfile.read(wav_file_path)
    # Tăng tốc độ phát âm thanh và phát
    sd.play(data, speed_factor * fs)
    sd.wait()
    # Xóa dữ liệu trong tệp WAV và MP3 sau khi phát xong
    with open(wav_file_path, 'wb') as wav_file:
        wav_file.truncate(0)
    with open(wav_file_path.replace('.wav', '.mp3'), 'wb') as mp3_file:
        mp3_file.truncate(0)
def run():
    wav_file_path = 'temp_audio1.wav'
    while True:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Đang nghe...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio, language="vi-VN")
            print("Đã nhận được: " + user_input)  
        except sr.UnknownValueError:
            print("Không nhận dạng được giọng nói.")
        if user_input == 'kết thúc':
            break
        
        max_matched_words,best_answer,question_in_data=compare(user_input,data)
        print("tra loi:",best_answer)
        tts = gTTS(best_answer, lang='vi')
        tts.save('output.mp3')
        play_wav(wav_file_path, speed_factor=1.0)
run()
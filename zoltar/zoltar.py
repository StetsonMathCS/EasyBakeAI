import os
import io
import hashlib
import pyaudio
import wave
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import texttospeech
from wit import Wit
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import cv2
import threading

# activity progression: wait -> listen -> camera -> wait
activity = 'wait'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "speech-in/speech.wav"

clarifai = ClarifaiApp(api_key=os.environ["CLARIFAI_KEY"])
clarifai_general_model = clarifai.models.get("aaa03c23b3724a16a56b629203edc62c") # general

speech = speech.SpeechClient()
speech_config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code='en-US')

tts = texttospeech.TextToSpeechClient()
voice = texttospeech.types.VoiceSelectionParams(
    language_code='en-US',
    name='en-US-Wavenet-D')
tts_audio_config = texttospeech.types.AudioConfig(
    speaking_rate=0.81,
    pitch=-7.60,
    audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)

wit_client = Wit(os.environ['WIT_TOKEN'])

cap = cv2.VideoCapture(0) # Capture video from camera

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)


p = pyaudio.PyAudio()

def record_speech():
    global p
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    frames = []

    print("* recording")
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")

    stream.stop_stream()
    stream.close()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def play_speech(fname):
    global p
    wf = wave.open(fname, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
    data = wf.readframes(CHUNK)
    while data != b'':
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()

def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = types.RecognitionAudio(content=content)
    response = speech.recognize(speech_config, audio)
    return response.results

def synthesize_text(text):
    # don't recreate speech we already have
    fname = 'speech-out/%s.wav' % hashlib.md5(text.encode()).hexdigest()
    if os.path.isfile(fname):
        return fname
    
    input_text = texttospeech.types.SynthesisInput(text=text)
    response = tts.synthesize_speech(input_text, voice, tts_audio_config)

    with open(fname, 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file %s' % fname)
    return fname

known_intents = {}
def infer_intent(text):
    global known_intents
    if text in known_intents:
        return known_intents[text]

    resp = wit_client.message(text)
    print('Response: {}'.format(resp))
    # {'_text': 'what is your name', 'entities': {'intent': [{'confidence': 0.97933293067394, 'value': 'what_is_name'}]}
    intents = []
    for intent in resp['entities']['intent']:
        intents.append(intent['value'])
    known_intents[text] = intents
    return intents

def get_camera_image():
    global activity
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if activity == 'grab':
                cv2.imwrite('camera-in/camera.jpg', frame)
                activity = 'wait'
                break
        else:
            activity = 'wait'
            return None

def find_objects_in_image():
    image = ClImage(file_obj=open('camera-in/camera.jpg', 'rb'))
    result = clarifai_general_model.predict([image])
    concepts = []
    for concept in result['outputs'][0]['data']['concepts']:
        concepts.append(concept['name'])
        print("%16s %.4f" % (concept['name'], concept['value']))

def do_zoltar():
    global activity
    while True:
        if activity == 'listen':
            record_speech()
            response = transcribe_file('speech-in/speech.wav')
            for result in response:
                text = result.alternatives[0].transcript
                print('Transcript: {}'.format(text))
                print(infer_intent(text))
                fname = synthesize_text(text)
                play_speech(fname)

            activity = 'camera'
            get_camera_image()
            print(find_objects_in_image())
            activity = 'wait'

def mouseclick(event):
    global activity
    if activity == 'wait':
        activity = 'listen'
    elif activity == 'camera':
        activity = 'grab'

thread = threading.Thread(target=do_zoltar)
thread.daemon = True
thread.start()

window = tk.Tk()
window.title("Zoltar")
window.geometry("1200x1600")
window.configure(background='grey')

path = "../../images/zotar.jpg"

img = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
panel.bind("<Button-1>", mouseclick)
window.mainloop()


p.terminate()
cap.release()
cv2.destroyAllWindows()


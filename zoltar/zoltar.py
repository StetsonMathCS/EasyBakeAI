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
import random
import calendar
from datetime import date
import audioop
from collections import deque
import math

# activity progression: wait -> listen -> camera -> wait
activity = 'wait'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
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

# adapted from: https://github.com/jeysonmc/python-google-speech-scripts/blob/master/stt_google.py

THRESHOLD = 300  # The threshold intensity that defines silence
                 # and noise signal (an int. lower than THRESHOLD is silence).

SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
                   # only silence is recorded. When this time passes the
                   # recording finishes and the file is delivered.

PREV_AUDIO = 0.5  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the phrase.

def listen_for_speech(threshold=THRESHOLD):
    """
    Listens to Microphone, extracts phrases from it and sends it to 
    Google's TTS service and returns response. a "phrase" is sound 
    surrounded by silence (according to threshold).
    """

    #Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Listening... ")
    audio2send = []
    cur_data = ''  # current chunk  of audio data
    rel = RATE/CHUNK
    slid_win = deque(maxlen=int(SILENCE_LIMIT * rel))
    #Prepend audio from 0.5 seconds before noise was detected
    prev_audio = deque(maxlen=int(PREV_AUDIO * rel))
    started = False

    while True:
        cur_data = stream.read(CHUNK)
        slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
        #print(slid_win[-1])
        #print(sum([x > THRESHOLD for x in slid_win])) 
        slid_win_sum = sum([x > THRESHOLD for x in slid_win]) 
        if(slid_win_sum > 4): # require at least 4 frames over threshold
            if(not started):
                print("Starting record of phrase")
                started = True
            audio2send.append(cur_data)
        elif (started is True):
            print("Finished")
            # The limit was reached, finish capture and deliver.
            frames = list(prev_audio)
            frames += audio2send
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            stream.stop_stream()
            stream.close()
            break
        else:
            prev_audio.append(cur_data)

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

def speak(text):
    fname = synthesize_text(text)
    play_speech(fname)

known_intents = {}
def infer_intent(text):
    global known_intents
    if text in known_intents:
        return known_intents[text]

    resp = wit_client.message(text)
    print('Response: {}'.format(resp))
    # {'_text': 'what is your name', 'entities': {'intent': [{'confidence': 0.97933293067394, 'value': 'what_is_name'}]}
    intents = []
    if 'intent' in resp['entities']:
        for intent in resp['entities']['intent']:
            intents.append(intent['value'])
        known_intents[text] = intents
    return intents

objects = []
def find_objects_in_image():
    global objects
    print('Grabbing camera image')
    if cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite('camera-in/camera.jpg', frame)
    image = ClImage(file_obj=open('camera-in/camera.jpg', 'rb'))
    print('Processing camera image')
    result = clarifai_general_model.predict([image])
    objects = []
    for concept in result['outputs'][0]['data']['concepts']:
        objects.append(concept['name'])
        print("%16s %.4f" % (concept['name'], concept['value']))

stalling_responses = [
    'Funny you should ask that question.',
    'I was just wondering a similar thing.',
    'While my answer may surprise you, it is indeed the truth.',
    'You surely ask from a source of great confusion.',
    'That question is new to me, let me take a moment.'
]

object_responses = [
    'I see {} is serving you well!',
    'You and your {} will make great strides.',
    'The world appreciates your {}.'
]

intent_responses = {
    'what_is_name': [
        'I am Zoltar!',
        'Zoltar my dear.'
    ],
    'what_day': [
        lambda: 'Today is {}, my dear.'.format(calendar.day_name[date.today().weekday()]),
        lambda: 'Why, it is {}, surely.'.format(calendar.day_name[date.today().weekday()])
    ]
}

nointent_responses = [
    'Now would be a good time to take up a new sport.',
    'The answer you seek is in the syllabus.'
]

def zoltar_response(query, intents, objects):
    responses = []
    if len(intents) > 0:
        intent = random.choice(intents)
        if intent in intent_responses:
            intent_resp = random.choice(intent_responses[intent])
            if callable(intent_resp):
                responses.append(intent_resp())
            else:
                responses.append(intent_resp)
    else:
        responses.append(random.choice(nointent_responses))

    if len(objects) > 0:
        obj = random.choice(objects)
        obj_resp = random.choice(object_responses)
        responses.append(obj_resp.format(obj))
    return ' '.join(responses)

def do_zoltar():
    global activity
    while True:
        if activity == 'listen':
            obj_thread = threading.Thread(target=find_objects_in_image)
            obj_thread.start()
            listen_for_speech()
            stalling_resp = random.choice(stalling_responses)
            stalling_thread = threading.Thread(target=speak, args=(stalling_resp,))
            stalling_thread.start()
            transcription = transcribe_file('speech-in/speech.wav')
            query = ''
            for t in transcription:
                query += '%s. ' % t.alternatives[0].transcript
            print('query:', query)
            intents = infer_intent(query)
            print('intents:', intents)
            obj_thread.join(timeout=3)
            print('objects:', objects)
            response = zoltar_response(query, intents, objects)
            print('response:', response)
            stalling_thread.join()
            speak(response)
            activity = 'wait'

def mouseclick(event):
    global activity
    if activity == 'wait':
        activity = 'listen'

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


import argparse
import os
import random
import shutil
import string
import sys
import wave
import pyaudio
from flask import Flask, request, send_file
from flask_cors import CORS

import noisereduce as nr
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.io import wavfile
from tqdm import tqdm
from googletrans import Translator


app = Flask(__name__)
CORS(app)

def listener(_port):
    app.run(host="0.0.0.0", port=_port, debug=False, threaded=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", help="Set port to receive audio from a client", dest="server")
    parser.add_argument(
        "-f", "--file", help="Path to audio file", dest="file",)
    parser.add_argument("-nr", "--noise-reduction", dest="noise", help="Noise reduction:\n" +
                                                                       "    there are two levels:\n" +
                                                                       "        level 1 - Basic noise reduction (recommended)\n" +
                                                                       "        level 2 - Massive noise reduction\n"
                        )
    parser.add_argument(
        "-o", "--output", help="Path to output file", dest="out", required=True)
    parser.add_argument("-iv", "--increase-volume", dest="iv", help="Increase volume:\n" +
                                                                    "    you have to provide a float " +
                                                                    "from 0 to 3 in the form int.dec")
    parser.add_argument("-l", "--language", dest="lang", help="Language (Default: en-EN)")
    parser.add_argument("-t", "--translate", dest="translate", help="Translate to language")

    return parser.parse_args()


folder_name = "audio-chunks"
args = get_args()

# Speech recognition object
r = sr.Recognizer()


def save_audio_to_wav(audio, filename):
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(audio.channel_count)
        wav_file.setsampwidth(audio.sample_width)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(audio.get_wav_data())


def translate_text(text, output_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=output_language).text
    return translated_text


def get_from_microphone():
    with sr.Microphone() as source:
        print("[+] Starting recording...")
        try:
            audio = r.listen(source, timeout=5)
        except Exception:
            return

        save_audio_to_wav(audio, 'temp.wav')


# Noisereduce library for massive noise reduction using spectral gates
def massive_noisereduction(audio_file):
    print("[+] Starting massive noise reduction...")

    # Random name for the new file
    random_string = ''.join(random.choices(
        string.ascii_letters + string.digits, k=8))
    file_name = random_string + '.wav'
    audio_file.export(file_name, format='wav')

    # Noise reduction
    rate, data = wavfile.read(file_name)
    orig_shape = data.shape
    data = np.reshape(data, (2, -1))
    reduced_noise = nr.reduce_noise(
        y=data, sr=rate, n_jobs=-1, stationary=True)

    # cleanup
    os.remove(file_name)

    file_name = "reduced_" + file_name
    wavfile.write(file_name, rate,
                  reduced_noise.reshape(orig_shape))
    sound = AudioSegment.from_file(file_name)

    # cleanup
    os.remove(file_name)

    print("[+] Noise reduction completed! :D")
    return sound


# Adding dbs to increase volume
def increase_volume(sound, level):
    print("[+] Incresing volume...")
    sound = sound + 10 * level
    print("[+] Volume increased! :D")
    return sound


def process_chunk(chunk_filename):
    with sr.AudioFile(chunk_filename) as source:

        if args.noise:
            if args.noise == 1:
                # Noise reduction
                r.adjust_for_ambient_noise(source, duration=2)

        audio_listened = r.record(source)
        # Convert to text
        try:
            if args.lang:
                language = args.lang
            else:
                language = "en-EN"
            text = r.recognize_google(audio_listened, language=language)
        except sr.UnknownValueError as e:
            pass
            # print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            return text


def transcribe_audio(path):

    print(args)
    if path is None:
        path = 'temp.wav'
        get_from_microphone()

    sound = AudioSegment.from_file(path)
    if args.iv:
        k = float(args.iv)
        if k <= 3 and k > 0:
            sound = increase_volume(sound, float(k))
        else:
            print("Bro, read the instructions!")
            sys.exit()

    if args.noise:
        if int(args.noise) == 2:
            sound = massive_noisereduction(sound)

    print("[+] Splitting in chunks...")

    # Split audio where silence is 1200ms or greater and get chunks (sometimes 700 and 700 is better)
    chunks = split_on_silence(
        sound, min_silence_len=1200, silence_thresh=sound.dBFS - 14, keep_silence=1000)
    print("[+] " + str(len(chunks)) + " chunks generated! :D")

    # Create folder to store audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    whole_text = ""

    print("[+] Processing chunks...")

    # Process each chunk
    for i, audio_chunk in enumerate(tqdm(chunks), start=1):

        # Export chunk and save in folder
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        chunk_silent = AudioSegment.silent(duration=5)
        audio_chunk = chunk_silent + audio_chunk + chunk_silent
        audio_chunk.export(chunk_filename, format="wav")

        # Recognize chunk
        text = process_chunk(chunk_filename)
        if text:
            whole_text += text

    print("[+] Transcription completed! :D")

    # cleanup
    shutil.rmtree(folder_name)

    # Return text for all chunks
    return whole_text

@app.route('/transcriber', methods=['GET'])
def get_page():
    return send_file('transcriber.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return "No file part"
    file = request.files['audio']
    file.save('temp.wav')
    result = transcribe_audio('temp.wav')
    if args.translate is not None:
        result = translate_text(result, args.translate)
    return result

if args.server is not None:
    listener(str(args.server))
else:
    result = transcribe_audio(args.file)
    if args.translate is not None:
        result = translate_text(result, args.translate)

    print(result, file=open(args.out, 'w'))

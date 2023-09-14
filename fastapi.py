from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import torchaudio
import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice

app = FastAPI()

# Initialize the TextToSpeech model
tts = TextToSpeech()

# Constants
VOICES = ["random", "train_atkins", "train_daws", "train_dotrice", "train_dreams", "train_empire", "train_grace", "train_kennard", "train_lescault", "train_mouse", "angie", "applejack", "daniel", "deniro", "emma", "freeman", "geralt", "halle", "jlaw", "lj", "mol", "myself", "pat", "pat2", "rainbow", "snakes", "tim_reynolds", "tom", "weaver", "william"]
DEFAULT_VOICE = "random"
PRESETS = ["ultra_fast", "fast", "standard", "high_quality"]
DEFAULT_PRESET = "fast"
DEFAULT_TEXT = "Hello, world!"

TORTOISE_SR_IN = 22050
TORTOISE_SR_OUT = 24000

class TTSForm(BaseModel):
    voice: str
    text: str
    model_preset: str

def chunk_audio(t: torch.Tensor, sample_rate: int, chunk_duration_sec: int) -> List[torch.Tensor]:
    duration = t.shape[1] / sample_rate
    num_chunks = 1 + int(duration / chunk_duration_sec)
    chunks = [t[:, (sample_rate * chunk_duration_sec * i):(sample_rate * chunk_duration_sec * (i + 1))] for i in range(num_chunks)]
    chunks = [chunk for chunk in chunks if chunk.shape[1] > 0]
    return chunks

def tts_main(voice_samples: List[torch.Tensor], text: str, model_preset: str) -> str:
    gen = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=None,
        preset=model_preset
    )
    torchaudio.save("generated.wav", gen.squeeze(0).cpu(), TORTOISE_SR_OUT)
    return "generated.wav"

def tts_from_preset(voice: str, text: str, model_preset: str):
    voice_samples, _ = load_voice(voice)
    return tts_main(voice_samples, text, model_preset)

def tts_from_files(files: List[UploadFile], do_chunk: bool, text: str, model_preset: str):
    voice_samples = []
    temp_files = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_files.append(temp_file.name)
            temp_file.write(file.file.read())
            audio = load_audio(temp_file.name, TORTOISE_SR_IN)
            if do_chunk:
                voice_samples.extend(chunk_audio(audio, TORTOISE_SR_IN, 10))
            else:
                voice_samples.append(audio)

    result = tts_main(voice_samples, text, model_preset)

    for temp_file in temp_files:
        os.remove(temp_file)

    return result

def tts_from_recording(recording: List[float], do_chunk: bool, text: str, model_preset: str):
    sample_rate, audio = recording
    norm_fix = 1

    if audio.dtype == int:
        norm_fix = 2**31
    elif audio.dtype == int:
        norm_fix = 2**15

    audio = torch.FloatTensor(audio) / norm_fix

    if len(audio.shape) > 1:
        audio = torch.mean(audio, axis=0).unsqueeze(0)

    audio = torchaudio.transforms.Resample(sample_rate, TORTOISE_SR_IN)(audio)

    if do_chunk:
        voice_samples = chunk_audio(audio, TORTOISE_SR_IN, 10)
    else:
        voice_samples = [audio]

    result = tts_main(voice_samples, text, model_preset)

    return result

@app.post("/tts/preset")
def tts_preset_endpoint(form: TTSForm):
    return tts_from_preset(form.voice, form.text, form.model_preset)

@app.post("/tts/files")
def tts_files_endpoint(
    files: List[UploadFile] = File(...),
    do_chunk: bool = Form(...),
    text: str = Form(...),
    model_preset: str = Form(...),
):
    return tts_from_files(files, do_chunk, text, model_preset)

@app.post("/tts/recording")
def tts_recording_endpoint(
    sample_rate: int = Form(...),
    audio: List[float] = Form(...),
    do_chunk: bool = Form(...),
    text: str = Form(...),
    model_preset: str = Form(...),
):
    return tts_from_recording([sample_rate, audio], do_chunk, text, model_preset)

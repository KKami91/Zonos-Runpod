import os
import base64
import runpod
import torch
import torchaudio
import tempfile
from io import BytesIO

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# Global variables
model = None

def initialize_model():
    global model
    if model is None:
        # Choose between transformer or hybrid model
        # model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    return model

def decode_base64_to_audio(base64_string):
    """Decode a base64 string to an audio file and return the path"""
    audio_bytes = base64.b64decode(base64_string)
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file.write(audio_bytes)
    temp_file.close()
    
    return temp_file.name

def audio_to_base64(audio_path):
    """Convert an audio file to base64 string"""
    with open(audio_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read())
    return encoded_string.decode('utf-8')

def handle_job(job):
    """Handle the Runpod job for text-to-speech generation"""
    job_input = job["input"]
    
    try:
        # Initialize model if not already done
        model = initialize_model()
        
        # Extract parameters from the job input
        text = job_input.get("text", "Hello, world!")
        language = job_input.get("language", "en-us")
        
        # Handle reference audio for voice cloning
        reference_audio_base64 = job_input.get("reference_audio")
        if not reference_audio_base64:
            return {"error": "Reference audio is required for voice cloning"}
        
        # Decode the base64 audio and load it
        reference_audio_path = decode_base64_to_audio(reference_audio_base64)
        wav, sampling_rate = torchaudio.load(reference_audio_path)
        
        # Clean up the temporary file
        os.unlink(reference_audio_path)
        
        # Create speaker embedding
        speaker = model.make_speaker_embedding(wav, sampling_rate)
        
        # Optional parameters
        speaking_rate = job_input.get("speaking_rate", 1.0)
        pitch_variation = job_input.get("pitch_variation", 1.0)
        max_frequency = job_input.get("max_frequency", 1.0)
        audio_quality = job_input.get("audio_quality", 1.0)
        emotion_happiness = job_input.get("emotion_happiness", 0.0)
        emotion_anger = job_input.get("emotion_anger", 0.0)
        emotion_sadness = job_input.get("emotion_sadness", 0.0)
        emotion_fear = job_input.get("emotion_fear", 0.0)
        
        # Create conditioning dictionary with all parameters
        cond_dict = make_cond_dict(
            text=text, 
            speaker=speaker, 
            language=language,
            speaking_rate=speaking_rate,
            pitch_variation=pitch_variation,
            max_frequency=max_frequency,
            audio_quality=audio_quality,
            emotion_happiness=emotion_happiness,
            emotion_anger=emotion_anger,
            emotion_sadness=emotion_sadness,
            emotion_fear=emotion_fear
        )
        
        conditioning = model.prepare_conditioning(cond_dict)
        
        # Generate audio codes
        codes = model.generate(conditioning)
        
        # Decode to audio
        wavs = model.autoencoder.decode(codes).cpu()
        
        # Save to a temporary file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
        
        # Convert to base64
        audio_base64 = audio_to_base64(output_path)
        
        # Clean up
        os.unlink(output_path)
        
        return {
            "audio": audio_base64,
            "sampling_rate": model.autoencoder.sampling_rate
        }
        
    except Exception as e:
        return {"error": str(e)}

# Initialize Runpod
runpod.serverless.start({"handler": handle_job})
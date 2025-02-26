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
    # 바이너리 데이터를 base64로 인코딩한 결과는 항상 ASCII 문자만 포함하므로 
    # ASCII로 디코딩해도 안전합니다. 하지만 호환성을 위해 'utf-8'을 유지합니다.
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
        speaking_rate = job_input.get("speaking_rate", 15.0)  # 기본값을 원본 함수와 맞춤
        pitch_std = job_input.get("pitch_std", 20.0)  # pitch_variation 대신 pitch_std 사용
        fmax = job_input.get("fmax", 22050.0)  # max_frequency 대신 fmax 사용
        
        # 감정 관련 파라미터 (전체 8개 감정 중 4개만 사용자 정의)
        # 기본 감정 배열 [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
        emotion = [0.0256] * 8  # 기본값
        emotion_idx = {
            "happiness": 0,
            "anger": 2,
            "sadness": 6,
            "fear": 7
        }
        
        # 사용자 정의 감정 값 적용
        emotion[emotion_idx["happiness"]] = job_input.get("emotion_happiness", 0.0256)
        emotion[emotion_idx["anger"]] = job_input.get("emotion_anger", 0.0256)
        emotion[emotion_idx["sadness"]] = job_input.get("emotion_sadness", 0.0256)
        emotion[emotion_idx["fear"]] = job_input.get("emotion_fear", 0.0256)
        
        # audio_quality 매개변수 사용하지 않음 (Zonos에 없음)
        
        # Create conditioning dictionary with correct parameters
        cond_dict = make_cond_dict(
            text=text, 
            speaker=speaker, 
            language=language,
            speaking_rate=speaking_rate,
            pitch_std=pitch_std,
            fmax=fmax,
            emotion=emotion
        )
        
        conditioning = model.prepare_conditioning(cond_dict)
        
        # Generate audio codes
        codes = model.generate(conditioning)
        
        # Decode to audio
        wavs = model.autoencoder.decode(codes).cpu()
        
        # Save to a temporary file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
        
        # Convert to base64...
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
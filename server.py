import os
import sys
import numpy as np
import torch
import warnings
import re
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from funasr import AutoModel

# Suppress PyTorch/Whisper warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
RATE = 16000
VAD_THRESHOLD = 0.5
MIN_SILENCE_DURATION_MS = 700
MAX_DURATION_S = 15

# --- Model Initialization ---
print("[INIT] Loading SenseVoice-Small model...")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"[INIT] Using device: {device}")

asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device=device,
    disable_update=True,
    disable_pbar=True
)

print("[INIT] Loading Silero VAD (ONNX) model...")
vad_model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=True
)

# --- Helper Functions ---
def format_sensevoice_output(raw_text):
    """Parse SenseVoice output to separate text from emotion/event tags."""
    # SenseVoice outputs tags like <|HAPPY|>, <|LAUGHTER|>, <|zh|>, <|en|>
    tags = re.findall(r"<\|[A-Z_]+\|>", raw_text)
    # Filter out language tags
    event_tags = [t for t in tags if t not in ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>"]]
    # Clean the text
    clean_text = re.sub(r"<\|[a-zA-Z_]+\|>", "", raw_text).strip()
    return clean_text, event_tags

# --- Server Setup ---
app = FastAPI()

# Serve static files (HTML)
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    response = FileResponse("static/index.html")
    # Disable caching for index.html to force browser update
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    
    # Session state
    audio_buffer = []
    state = "SILENCE"
    silence_counter = 0
    chunks_since_last_transcribe = 0
    
    # Parameters for VAD state machine
    chunk_size = 512 # We expect client to send ~512 samples at 16kHz (32ms)
    partial_interval_chunks = int(0.5 * RATE / chunk_size) # 500ms
    min_silence_chunks = int((MIN_SILENCE_DURATION_MS / 1000) * RATE / chunk_size)
    max_chunks = int(MAX_DURATION_S * RATE / chunk_size)
    
    try:
        while True:
            # Receive audio chunk (bytes)
            data = await websocket.receive_bytes()
            
            # Convert bytes to float32 numpy array
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Debug: Print RMS (volume) periodically
            rms = np.sqrt(np.mean(audio_np**2))
            
            # Print first few chunks to verify connection
            # if silence_counter < 5 and rms > 0.0:
            #    print(f"[WS DEBUG] Chunk size: {len(audio_np)}, RMS: {rms:.4f}")

            if rms > 0.005: # Lower threshold to catch even faint sounds
                # Only print if there is some sound to avoid spam
                # print(f"[WS] Audio chunk received, RMS: {rms:.4f}")
                pass
            
            # VAD Inference
            # Silero VAD strictly requires chunk size of 512, 1024, or 1536 samples for 16k rate
            # Our client is sending 2048 samples, which is 128ms
            # We need to process it in chunks of 512
            
            # Split incoming audio into 512-sample chunks for VAD
            vad_chunk_size = 512
            
            # Determine overall speech status for this large chunk
            # If ANY 512-subchunk is speech, we consider the whole 2048 chunk as contributing to speech
            # But for state machine precision, we should ideally feed 512 chunks one by one
            
            num_subchunks = len(audio_np) // vad_chunk_size
            remainder = len(audio_np) % vad_chunk_size
            
            # If we have remainder, we might lose a bit, but 2048 % 512 == 0
            
            is_speech_frame = False
            max_confidence = 0.0
            
            for i in range(num_subchunks):
                start = i * vad_chunk_size
                end = start + vad_chunk_size
                sub_chunk = audio_np[start:end]
                
                audio_tensor = torch.from_numpy(sub_chunk)
                try:
                    # VAD expects (batch, time) or just (time)
                    # For onnx, usually flat tensor is fine
                    confidence = vad_model(audio_tensor, RATE).item()
                    max_confidence = max(max_confidence, confidence)
                except Exception as e:
                    # print(f"[VAD ERROR] {e}")
                    confidence = 0.0
            
            # Use max confidence from subchunks to drive state machine
            confidence = max_confidence
            
            # Debug VAD
            # if confidence > 0.1:
            #    print(f"[VAD] RMS: {rms:.4f}, Conf: {confidence:.2f}, State: {state}")
            
            # Force print for debugging (Commented out for production)
            # if rms > 0.005 or confidence > 0.1:
            #      # Throttled print
            #      if silence_counter % 5 == 0: 
            #          print(f"[VAD] RMS: {rms:.4f}, Conf: {confidence:.2f}, State: {state}")
            
            # --- State Machine Logic (Same as main.py) ---
            
            if state == "SILENCE":
                if confidence >= VAD_THRESHOLD:
                    state = "SPEECH"
                    audio_buffer = [audio_np]
                    silence_counter = 0
                    chunks_since_last_transcribe = 0
                    # Notify client: Speech Started
                    await websocket.send_json({"type": "status", "content": "LISTENING"})
            
            elif state == "SPEECH":
                audio_buffer.append(audio_np)
                chunks_since_last_transcribe += 1
                
                if confidence < VAD_THRESHOLD:
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                # --- Partial Transcription ---
                if chunks_since_last_transcribe >= partial_interval_chunks:
                    chunks_since_last_transcribe = 0
                    # Run inference on current buffer
                    speech_data = np.concatenate(audio_buffer)
                    
                    # Run SenseVoice
                    res = asr_model.generate(
                        input=speech_data,
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=60,
                        merge_vad=True,
                        merge_length_s=15
                    )
                    
                    text = ""
                    if res and isinstance(res, list) and len(res) > 0:
                        text = res[0].get("text", "")
                    
                    clean_text, tags = format_sensevoice_output(text)
                    
                    if clean_text:
                        await websocket.send_json({
                            "type": "partial",
                            "text": clean_text,
                            "tags": tags
                        })
                
                # --- End of Speech Check ---
                if silence_counter >= min_silence_chunks or len(audio_buffer) >= max_chunks:
                    state = "SILENCE"
                    
                    # Final Inference
                    speech_data = np.concatenate(audio_buffer)
                    res = asr_model.generate(
                        input=speech_data,
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=60,
                        merge_vad=True,
                        merge_length_s=15
                    )
                    
                    text = ""
                    if res and isinstance(res, list) and len(res) > 0:
                        text = res[0].get("text", "")
                        
                    clean_text, tags = format_sensevoice_output(text)
                    
                    await websocket.send_json({
                        "type": "final",
                        "text": clean_text,
                        "tags": tags
                    })
                    
                    # Reset
                    audio_buffer = []
                    silence_counter = 0
                    chunks_since_last_transcribe = 0
                    
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

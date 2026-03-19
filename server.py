import os
import sys

# Fix for OMP error on macOS - Must be before importing numpy/torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import warnings
import re
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from fastapi.responses import FileResponse
from funasr import AutoModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import psutil
import threading

# Load environment variables
load_dotenv()

# Fix for OMP error on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Suppress PyTorch/Whisper warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
RATE = 16000
VAD_THRESHOLD = 0.5
MIN_SILENCE_DURATION_MS = 700
MAX_DURATION_S = 15
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# --- DeepSeek Client ---
deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
else:
    print("[WARN] DEEPSEEK_API_KEY not found. AI Analysis feature will be disabled.")

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

# Estimate Model Info (SenseVoice-Small)
MODEL_INFO = {
    "name": "SenseVoice-Small",
    "params": "50M",
    "memory": " ~300 MB",
    "device": device.upper()
}

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
    tags = re.findall(r"<\|[A-Z_]+\|>", raw_text)
    event_tags = [t for t in tags if t not in ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>"]]
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
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response

# Global state
active_websockets = set()
session_transcript = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    
    # State for this connection
    audio_buffer = []
    pre_roll_buffer = [] # Buffer to keep a few chunks before speech starts
    state = "SILENCE"
    silence_counter = 0
    last_partial_time = 0
    
    chunk_size = 512 # Approx 32ms
    min_silence_chunks = int((MIN_SILENCE_DURATION_MS / 1000) * (RATE / chunk_size))
    max_chunks = int(MAX_DURATION_S * (RATE / chunk_size))
    pre_roll_chunks = int(300 / 1000 * RATE / chunk_size) # 300ms pre-roll
    
    print(f"[WS] Client connected. Active: {len(active_websockets)}")
    
    try:
        while True:
            # Receive bytes (0x00+Audio or 0x01+Video) or Text (JSON)
            message = await websocket.receive()
            
            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "analyze":
                    if not deepseek_client:
                        await websocket.send_json({"type": "error", "content": "DeepSeek API Key not configured."})
                        continue
                        
                    full_text = "\n".join(session_transcript)
                    if not full_text:
                        await websocket.send_json({"type": "error", "content": "No transcript to analyze yet."})
                        continue
                        
                    await websocket.send_json({"type": "analysis", "content": "Analyzing..."})
                    
                    try:
                        response = await deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Summarize the following meeting transcript."},
                                {"role": "user", "content": full_text}
                            ],
                            stream=False
                        )
                        analysis = response.choices[0].message.content
                        await websocket.send_json({"type": "analysis", "content": analysis})
                    except Exception as e:
                        await websocket.send_json({"type": "error", "content": f"DeepSeek Error: {str(e)}"})
                        
            elif "bytes" in message:
                data = message["bytes"]
                
                if len(data) > 0:
                    msg_type = data[0]
                    payload = data[1:]
                    
                    if msg_type == 1: # Video Frame
                        # Save latest frame
                        try:
                            with open("static/live.jpg", "wb") as f:
                                f.write(payload)
                        except Exception as e:
                            print(f"[VIDEO] Error saving frame: {e}")
                            
                    elif msg_type == 0: # Audio Chunk
                        # Convert bytes to numpy array (float32, normalized)
                        audio_int16 = np.frombuffer(payload, dtype=np.int16)
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0
                        
                        # Always append to pre_roll buffer to keep a short history
                        pre_roll_buffer.append(audio_float32)
                        if len(pre_roll_buffer) > pre_roll_chunks:
                            pre_roll_buffer.pop(0)
                            
                        if state == "SPEECH":
                            audio_buffer.append(audio_float32)
                        
                        # VAD Logic requires 512 samples for 16000Hz
                        vad_audio = audio_float32
                        
                        if len(vad_audio) == 512:
                            try:
                                # VAD model is loaded on CPU by default (especially ONNX version)
                                # So we must keep the input tensor on CPU, NOT move it to MPS/CUDA
                                vad_tensor = torch.from_numpy(vad_audio).float()
                                
                                speech_prob = vad_model(vad_tensor, RATE).item()
                                
                                if speech_prob > VAD_THRESHOLD:
                                    if state == "SILENCE":
                                        state = "SPEECH"
                                        # When speech starts, seed the buffer with the pre-roll to catch the beginning of the word
                                        audio_buffer = list(pre_roll_buffer)
                                        await websocket.send_json({"type": "vad", "status": "speech_start"})
                                    silence_counter = 0
                                else:
                                    if state == "SPEECH":
                                        silence_counter += 1
                            except Exception as e:
                                print(f"[VAD] Error: {e}")
                                
                        # Check if we should transcribe
                        should_transcribe_final = False
                        should_transcribe_partial = False
                        
                        if state == "SPEECH":
                            current_time = time.time()
                            # Do a partial transcription every 0.5 seconds if we have enough audio
                            if current_time - last_partial_time > 0.5 and len(audio_buffer) > pre_roll_chunks + 10:
                                should_transcribe_partial = True
                                last_partial_time = current_time
                                
                            if silence_counter > min_silence_chunks:
                                state = "SILENCE"
                                await websocket.send_json({"type": "vad", "status": "speech_end"})
                                should_transcribe_final = True
                            elif len(audio_buffer) > max_chunks:
                                # Force transcribe if it's too long to prevent latency and memory issues
                                should_transcribe_final = True
                                
                        if (should_transcribe_final or should_transcribe_partial) and len(audio_buffer) > 0:
                            current_audio = np.concatenate(audio_buffer)
                            
                            try:
                                res = asr_model.generate(
                                    input=current_audio,
                                    cache={},
                                    language="zh", # Set to 'zh' explicitly to improve Mandarin accuracy
                                    use_itn=True,
                                    batch_size_s=60,
                                )
                                text = res[0]["text"]
                                clean_text, events = format_sensevoice_output(text)
                                
                                if clean_text:
                                    msg_type = "result" if should_transcribe_final else "partial_result"
                                    if should_transcribe_final:
                                        print(f"[ASR] Final Result: {clean_text}")
                                        session_transcript.append(clean_text)
                                    
                                    # Broadcast result
                                    response_msg = {
                                        "type": msg_type,
                                        "text": clean_text,
                                        "events": events
                                    }
                                    for ws in list(active_websockets):
                                        try:
                                            await ws.send_json(response_msg)
                                        except:
                                            pass
                            except Exception as e:
                                print(f"[ASR] Error: {e}")
                            
                            if should_transcribe_final:
                                audio_buffer = []
                                last_partial_time = 0
                                if state == "SPEECH":
                                    silence_counter = 0
                                
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        active_websockets.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

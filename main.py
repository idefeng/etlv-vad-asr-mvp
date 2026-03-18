import pyaudio
import numpy as np
import threading
import queue
import time
import torch
import warnings
from funasr import AutoModel
import re

# Suppress PyTorch/Whisper warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)

class RealTimeSpeechProcessor:
    def __init__(self, vad_threshold=0.5, min_silence_duration_ms=700, max_duration_s=15):
        # Audio configuration
        self.RATE = 16000
        self.CHUNK = 512  # Optimal chunk size for Silero VAD at 16kHz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        
        # Initialization
        print("[INIT] Loading SenseVoice-Small model...")
        # device="cpu", "cuda", or "mps" for Mac M-series
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            
        print(f"[INIT] Using device: {device}")
        
        # Load SenseVoice-Small from ModelScope/FunASR
        # trust_remote_code=True is required for custom model implementations
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device=device,
            disable_update=True,
            disable_pbar=True
        )
        
        print("[INIT] Loading Silero VAD (ONNX) model...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        self.asr_queue = queue.Queue()
        self.is_running = False
        
        # State machine variables
        self.state = "SILENCE"  # "SILENCE" or "SPEECH"
        self.audio_buffer = []
        self.silence_counter = 0
        self.threshold = vad_threshold
        
        # Streaming/Incremental transcription variables
        self.chunks_since_last_transcribe = 0
        self.partial_interval_chunks = int(0.5 * self.RATE / self.CHUNK) # Send partial update every 500ms
        self.min_silence_chunks = int((min_silence_duration_ms / 1000) * self.RATE / self.CHUNK)
        self.max_chunks = int(max_duration_s * self.RATE / self.CHUNK)

    def start(self):
        """Start the audio capture and processing loops."""
        self.is_running = True
        
        # Start ASR worker thread to prevent blocking audio stream
        self.asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
        self.asr_thread.start()
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("\n[READY] Start speaking...")
        self._print_status("[LISTENING]")
        
        try:
            self._process_loop()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop processing and cleanup."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("\n[STOPPED] Audio capture stopped.")

    def _print_status(self, status):
        # Overwrite current line
        print(f"\r\033[K{status} ", end="", flush=True)

    def _process_loop(self):
        """Main audio capture and VAD processing loop."""
        while self.is_running:
            try:
                # Non-blocking read (exception_on_overflow=False handles minor drops gracefully)
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            except IOError:
                continue
            
            # Convert Int16 buffer to Float32 [-1.0, 1.0] for VAD model
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Inference VAD model
            # Note: For Silero VAD, we pass the raw tensor of shape (CHUNK,)
            audio_tensor = torch.from_numpy(audio_np)
            confidence = self.vad_model(audio_tensor, self.RATE).item()
            
            # State Machine Logic
            if self.state == "SILENCE":
                if confidence >= self.threshold:
                    self.state = "SPEECH"
                    self._print_status("[SPEAKING]")
                    self.audio_buffer = [audio_np]
                    self.silence_counter = 0
                    self.chunks_since_last_transcribe = 0
            
            elif self.state == "SPEECH":
                self.audio_buffer.append(audio_np)
                self.chunks_since_last_transcribe += 1
                
                if confidence < self.threshold:
                    self.silence_counter += 1
                else:
                    self.silence_counter = 0
                    
                # --- Streaming / Partial Update Logic ---
                # Every ~500ms, if the ASR worker is idle, send a partial buffer for "smooth" real-time typing effect
                if self.chunks_since_last_transcribe >= self.partial_interval_chunks:
                    self.chunks_since_last_transcribe = 0
                    if self.asr_queue.empty(): # Only push if queue is empty to avoid processing lag
                        speech_data = np.concatenate(self.audio_buffer)
                        self.asr_queue.put((speech_data, False)) # False means is_final=False
                # ----------------------------------------
                    
                # Check End of Speech: Silence exceeds threshold OR max duration reached
                if self.silence_counter >= self.min_silence_chunks or len(self.audio_buffer) >= self.max_chunks:
                    self.state = "SILENCE"
                    
                    # Prepare audio array for ASR and enqueue
                    speech_data = np.concatenate(self.audio_buffer)
                    
                    # Clear any pending partial updates so we prioritize the final transcription
                    while not self.asr_queue.empty():
                        try: self.asr_queue.get_nowait()
                        except: pass
                        
                    self.asr_queue.put((speech_data, True)) # True means is_final=True
                    
                    # Reset buffers
                    self.audio_buffer = []
                    self.silence_counter = 0
                    self.chunks_since_last_transcribe = 0

    def _format_sensevoice_output(self, raw_text):
        """Parse SenseVoice output to separate text from emotion/event tags."""
        # SenseVoice outputs tags like <|HAPPY|>, <|LAUGHTER|>, <|zh|>, <|en|>
        
        # 1. Extract emotion/event tags
        tags = re.findall(r"<\|[A-Z_]+\|>", raw_text)
        
        # Filter out language tags (e.g., <|zh|>, <|en|>, <|yue|>)
        event_tags = [t for t in tags if t not in ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>"]]
        
        # 2. Clean the text (remove all tags)
        clean_text = re.sub(r"<\|[a-zA-Z_]+\|>", "", raw_text).strip()
        
        return clean_text, event_tags

    def _asr_worker(self):
        """Dedicated thread for running SenseVoice inferences asynchronously."""
        while self.is_running:
            try:
                # Wait for audio chunks from the main thread
                audio_data, is_final = self.asr_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                # SenseVoice requires specific input format.
                # Usually it takes a file path or a torch tensor.
                # Let's convert numpy to torch tensor.
                # Note: SenseVoice expects input shape (batch, time) or (time,)
                # Since we are doing single stream, we wrap it.
                
                # FunASR generate accepts: input=...
                # We can pass the numpy array directly if supported, or wrap in list.
                
                # IMPORTANT: FunASR expects 16kHz audio. Our input is already 16kHz float32.
                # However, FunASR models usually expect waveform to be unnormalized or normalized?
                # Usually standard is normalized float32 [-1, 1].
                
                # Run inference
                # language="auto" allows it to detect zh/en/yue/ja/ko
                # use_itn=True enables inverse text normalization (e.g. "one hundred" -> "100")
                res = self.asr_model.generate(
                    input=audio_data,
                    cache={},
                    language="auto", 
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15
                )
                
                # Result is usually a list of dicts: [{'text': '...'}]
                if res and isinstance(res, list) and len(res) > 0:
                    raw_text = res[0].get("text", "")
                else:
                    raw_text = ""
                
                clean_text, event_tags = self._format_sensevoice_output(raw_text)
                
                # Construct display string
                display_str = clean_text
                if event_tags:
                    # Append tags nicely, e.g. "Hello world [HAPPY] [LAUGHTER]"
                    tag_str = " ".join([t.replace("<|", "[").replace("|>", "]") for t in event_tags])
                    display_str = f"{clean_text} \033[93m{tag_str}\033[0m" # Yellow color for tags

                if display_str:
                    if is_final:
                        # Clear current line and print final result, followed by a newline
                        print(f"\r\033[K[RESULT] {display_str}")
                    else:
                        # Truncate preview to avoid line wrapping which breaks \r refresh
                        preview_str = display_str
                        max_preview_len = 60
                        if len(preview_str) > max_preview_len:
                            preview_str = "..." + preview_str[-max_preview_len:]
                        
                        # Print partial result on the same line to create a real-time typing effect
                        print(f"\r\033[K[SPEAKING] {preview_str}...", end="", flush=True)
                
                # Resume listening UI if the state is back to SILENCE and we just finished the final chunk
                if is_final and self.state == "SILENCE":
                    self._print_status("[LISTENING]")
                    
            except Exception as e:
                print(f"\n[ERROR] Inference failed: {e}")
                if is_final:
                    self._print_status("[LISTENING]")
                
            self.asr_queue.task_done()

if __name__ == "__main__":
    processor = RealTimeSpeechProcessor(
        vad_threshold=0.5, 
        min_silence_duration_ms=700,
        max_duration_s=15
    )
    processor.start()

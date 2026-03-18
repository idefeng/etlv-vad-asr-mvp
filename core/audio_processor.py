import queue
import sounddevice as sd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=512, channels=1):
        """
        初始化音频处理器。
        :param sample_rate: 采样率，默认为 16000Hz (Silero VAD 推荐)。
        :param chunk_size: 每个音频块的大小（采样点数）。
        :param channels: 通道数，默认 1 (Mono)。
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.stream = None

    def _audio_callback(self, indata, frames, time, status):
        """
        音频采集回调函数，将采集到的音频块放入队列。
        """
        if status:
            logger.warning(f"Audio Stream Status: {status}")
        # 将采集到的原始数据推入队列（indata 是 numpy 数组）
        # Silero VAD 通常需要 float32 格式，范围 [-1, 1]
        self.audio_queue.put(indata.copy().flatten())

    def start_recording(self):
        """
        开启音频流录制。
        """
        logger.info(f"Starting audio stream: {self.sample_rate}Hz, channels={self.channels}")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size,
            dtype='float32'
        )
        self.stream.start()

    def stop_recording(self):
        """
        停止音频流录制。
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped.")

    def get_chunks(self):
        """
        迭代器，持续产出音频块。
        """
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

if __name__ == "__main__":
    # 简单的本地测试逻辑
    import time
    processor = AudioProcessor(chunk_size=1536) # Silero VAD 常用 512, 1024, 1536
    try:
        processor.start_recording()
        print("Recording... Press Ctrl+C to stop.")
        for i, chunk in enumerate(processor.get_chunks()):
            if i % 10 == 0:
                print(f"Captured chunk {i}, shape: {chunk.shape}, max_val: {np.max(chunk):.4f}")
            if i > 100: # 测试捕获 100 个块后自动停止
                break
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop_recording()

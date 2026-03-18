import os
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VADHandler:
    def __init__(self, model_path=None, sample_rate=16000, threshold=0.5, min_silence_duration_ms=500):
        """
        初始化 VAD 处理器。
        :param model_path: Silero VAD 模型路径（可选）。
        :param sample_rate: 采样率。
        :param threshold: 语音判定阈值 (0.0 - 1.0)。
        :param min_silence_duration_ms: 判定语音结束前的最小静音时长（毫秒）。
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_samples = (min_silence_duration_ms * sample_rate) // 1000
        
        # 加载 Silero VAD 模型
        # 使用 torch.hub 是最简单的方式，会自动处理模型下载与加载
        # 为了端侧性能，如果已安装 onnxruntime，silero-vad 内部会尝试使用
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          onnx=True) # 使用 ONNX 加速
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = utils
        
        # 状态机变量
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_counter = 0

    def process_chunk(self, chunk):
        """
        处理单块音频，返回当前状态。
        :param chunk: numpy 数组或 torch 张量。
        :return: (is_speech_ended, full_audio_segment)
        """
        # 转换为 torch.Tensor 并增加维度 (batch=1)
        if isinstance(chunk, np.ndarray):
            tensor_chunk = torch.from_numpy(chunk).float()
        else:
            tensor_chunk = chunk
            
        # 获取语音概率
        with torch.no_grad():
            speech_prob = self.model(tensor_chunk, self.sample_rate).item()

        if speech_prob > self.threshold:
            self.silence_counter = 0
            if not self.is_speaking:
                self.is_speaking = True
                logger.debug("Speech Started")
            self.speech_buffer.append(chunk)
            return False, None
        else:
            if self.is_speaking:
                self.silence_counter += len(chunk)
                self.speech_buffer.append(chunk)

                # 超过最小静音时长，判定语音结束
                if self.silence_counter >= self.min_silence_samples:
                    self.is_speaking = False
                    logger.debug("Speech Ended")
                    full_segment = np.concatenate(self.speech_buffer)
                    self.speech_buffer = []
                    self.silence_counter = 0
                    return True, full_segment
            return False, None

    def reset(self):
        """重置状态机"""
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_counter = 0

if __name__ == "__main__":
    # 模拟测试逻辑
    handler = VADHandler()
    print("VAD model loaded successfully.")
    # 构造一段模拟信号（先静音，后噪音/语音，再静音）
    fake_chunk = np.random.uniform(-0.01, 0.01, 512).astype(np.float32)
    ended, segment = handler.process_chunk(fake_chunk)
    print(f"Prob test with silence: ended={ended}")

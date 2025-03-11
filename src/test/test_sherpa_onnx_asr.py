import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from close_llm_agent.asr.sherpa_onnx_asr import VoiceRecognition

# Dummy stream 实现，transcribe_np 会调用它的 accept_waveform 方法和读取 result.text 属性
class DummyStream:
    def __init__(self):
        class Result:
            text = "dummy transcription"
        self.result = Result()

    def accept_waveform(self, sample_rate, audio):
        # 模拟接受语音数据（这里不做任何处理）
        pass

# Dummy recognizer 实现，创建 stream 后 decode_streams 就不做任何操作
class DummyRecognizer:
    def create_stream(self):
        return DummyStream()

    def decode_streams(self, streams):
        # 模拟解码，不做任何处理，DummyStream 已经固定返回 "dummy transcription"
        pass

# 利用 monkeypatch，将 VoiceRecognition._create_recognizer 替换为返回 DummyRecognizer 的 lambda 函数
@pytest.fixture(autouse=True)
def patch_recognizer(monkeypatch):
    monkeypatch.setattr(VoiceRecognition, "_create_recognizer", lambda self: DummyRecognizer())

def test_transcribe_np():
    vr = VoiceRecognition()
    # 构造一个 dummy 的音频 numpy array, 比如 1 秒的静音 16000 个采样点
    dummy_audio = np.zeros(16000, dtype=np.float32)
    transcription = vr.transcribe_np(dummy_audio)
    assert transcription == "dummy transcription"
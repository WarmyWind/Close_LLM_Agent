import asyncio
import numpy as np
import pytest
import torch
import websockets
from close_llm_agent.vad.silero import VADEngine, SileroVADConfig, StateMachine, vad_main
# File: src/close-llm-agent/vad/test_silero.py



# Test __init__ functionality for VADEngine
def test_vade_engine_init():
    engine = VADEngine(
        orig_sr=16000,
        target_sr=16000,
        prob_threshold=0.4,
        db_threshold=60,
        required_hits=3,
        required_misses=24,
        smoothing_window=5,
    )
    assert isinstance(engine.config, SileroVADConfig)
    # Default target_sr=16000 yields window_size_samples == 512
    assert engine.window_size_samples == 512
    assert isinstance(engine.state, StateMachine)

# Dummy VAD model that always returns 0.5 probability.
def dummy_vad(chunk, sr):
    return torch.tensor(0.5)

@pytest.mark.asyncio
async def test_vad_service(monkeypatch):
    # Monkey-patch load_silero_vad to return our dummy VAD model.
    monkeypatch.setattr("close_llm_agent.vad.silero.load_silero_vad", lambda: dummy_vad)

    # Start the vad server as a background task.
    server_task = asyncio.create_task(vad_main())
    # Allow some time for server to start.
    await asyncio.sleep(1)

    try:
        # Connect to the websocket server.
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            # Build a dummy binary message.
            # Create a list of 512 float zeros and convert to bytes.
            dummy_audio = [0.0] * 512
            # Convert list to bytes via numpy array conversion to simulate audio data.
            audio_np = np.array(dummy_audio, dtype=np.float32)
            dummy_bytes = audio_np.tobytes()
            await websocket.send(dummy_bytes)
            # Since the server does not respond, we simply ensure no error occurs.
            await asyncio.sleep(0.5)
    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task
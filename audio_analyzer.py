"""
Audio analysis module for bird song recognition.
"""
import logging
import wave
from pathlib import Path

import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Analyze audio clips for bird song recognition."""

    def __init__(self, model_path='', labels=None, sample_rate=16000, sample_seconds=4, device_index=None):
        self.model_path = Path(model_path) if model_path else None
        self.labels = labels or []
        self.sample_rate = sample_rate
        self.sample_seconds = sample_seconds
        self.device_index = int(device_index) if str(device_index).isdigit() else None
        self.session = None
        self.input_name = None
        self.input_size = (128, 128)
        self._load()

    def _load(self):
        if not self.model_path or not self.model_path.exists():
            logger.info("Audio model not configured or missing")
            return
        if ort is None:
            logger.warning("onnxruntime not installed - audio model disabled")
            return
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        shape = input_info.shape
        if len(shape) >= 4 and shape[2] and shape[3]:
            self.input_size = (int(shape[3]), int(shape[2]))
        logger.info(f"Loaded audio model: {self.model_path}")

    def is_ready(self):
        return self.session is not None

    def record_clip(self):
        if sd is None:
            logger.info("sounddevice not installed - audio capture disabled")
            return None
        duration = self.sample_seconds
        frames = int(self.sample_rate * duration)
        try:
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, device=self.device_index)
            sd.wait()
            return recording.flatten()
        except Exception as exc:
            logger.error(f"Audio capture failed: {exc}")
            return None

    def load_wav(self, wav_path):
        path = Path(wav_path)
        if not path.exists():
            return None, None
        with wave.open(str(path), 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio = audio / (np.max(np.abs(audio)) + 1e-6)
        return audio, sample_rate

    def _spectrogram(self, audio, sample_rate):
        if audio is None or len(audio) == 0:
            return None
        hop = int(sample_rate * 0.01)
        win = int(sample_rate * 0.025)
        frames = 1 + (len(audio) - win) // hop if len(audio) > win else 1
        window = np.hanning(win)
        spec = []
        for i in range(frames):
            start = i * hop
            segment = audio[start:start + win]
            if len(segment) < win:
                segment = np.pad(segment, (0, win - len(segment)))
            spectrum = np.fft.rfft(segment * window)
            spec.append(np.abs(spectrum))
        spec = np.array(spec).T
        spec = np.log(spec + 1e-6)
        return spec

    def _prepare_input(self, audio, sample_rate):
        spec = self._spectrogram(audio, sample_rate)
        if spec is None:
            return None
        spec_resized = cv2.resize(spec, self.input_size)
        normalized = (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min() + 1e-6)
        tensor = normalized.astype(np.float32)[None, None, ...]
        return tensor

    def analyze(self, audio, sample_rate):
        if not self.is_ready():
            return {
                'species': 'Unknown Bird Song',
                'confidence': 0.0,
                'notes': 'Audio model not configured'
            }

        input_tensor = self._prepare_input(audio, sample_rate)
        if input_tensor is None:
            return {
                'species': 'Unknown Bird Song',
                'confidence': 0.0,
                'notes': 'Audio clip empty'
            }

        outputs = self.session.run(None, {self.input_name: input_tensor})
        scores = outputs[0][0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        species = self.labels[top_idx] if self.labels and top_idx < len(self.labels) else f"Class {top_idx}"
        return {
            'species': species,
            'confidence': confidence,
            'notes': 'Audio model inference'
        }


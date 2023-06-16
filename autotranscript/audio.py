import os
from warnings import warn

import numpy as np
import torch
import ffmpeg

SAMPLE_RATE = 16000

class AudioProcessor:
    """
    Audio Processor using PyTorchaudio instead of PyDub
    """
    
    def __init__(self, waveform: torch.Tensor, sr : torch.Tensor) -> None:
        """
        Initialise audio processor
        :param waveform: waveform
        :param sr: sample rate
        """
        self.waveform = waveform
        self.sr = sr
        
        if not isinstance(self.sr, int):
            raise ValueError("Sample rate should be a single value of type int," \
                             f"not {len(self.sr)} and type {type(self.sr)}")
        
    @classmethod
    def from_file(cls, file: str, *args, **kwargs) -> 'AudioProcessor':
        """
        Load audio file
        :param file: audio file
        :return: AudioProcessor
        """
        
        audio, sr = cls.load_audio(file , *args, **kwargs)

        audio = torch.from_numpy(audio)
        
        return cls(audio, sr)
    
    
    def cut(self, start: float, end: float) -> torch.Tensor:
        """
        Cut audio file
        :param start: start time in seconds
        :param end: end time in seconds
        :return: AudioProcessor
        """
        
        if isinstance(start, float):
            start = torch.Tensor([start])
        if isinstance(end, float):
            end = torch.Tensor([end])
        
        sr = torch.Tensor([self.sr])
            
        start = int(start * sr)
        end = torch.ceil(end * sr)
        
        return self.waveform[start:end.to(int)]

    @staticmethod
    def load_audio(file: str, sr: int = SAMPLE_RATE):
        """
        Open an audio file and read as mono waveform, resampling as necessary

        Changed from original function at whisper.audio.load_audio to ensure compatibility
        with pyannote.audio
        Parameters
        ----------
        file: str
            The audio file to open

        sr: int
            The sample rate to resample the audio if necessary

        Returns
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """
        try:
            # This launches a subprocess to decode audio while down-mixing 
            # and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le",
                        ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"],
                     capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        
        return out , sr
    
    def __repr__(self) -> str:
        return f'TorchAudioProcessor(waveform={len(self.waveform)}, sr={int(self.sr)})'
    
    def __str__(self) -> str:
        return f'TorchAudioProcessor(waveform={len(self.waveform)}, sr={int(self.sr)})'

    
if __name__ == "__main__":
    
    print("Testing AudioProcessor")
    print(AudioProcessor.from_file("tests/test.wav"))
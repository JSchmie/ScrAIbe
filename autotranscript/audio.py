"""
Audio Processor Module
=======================

This module provides the AudioProcessor class, utilizing PyTorchaudio for handling audio files.
It includes functionalities to load, cut, and manage audio waveforms, offering efficient and
flexible audio processing.

Available Classes:
- AudioProcessor: Processes audio waveforms and provides methods for loading, 
                    cutting, and handling audio.

Usage:
    from .audio_import AudioProcessor

    processor = AudioProcessor.from_file("path/to/audiofile.wav")
    cut_waveform = processor.cut(start=1.0, end=5.0)

Constants:
- SAMPLE_RATE (int): Default sample rate for processing.
- NORMALIZATION_FACTOR (float): Normalization factor for audio waveform.
"""

from subprocess import CalledProcessError, run
import numpy as np
import torch

SAMPLE_RATE = 16000
NORMALIZATION_FACTOR = 32768.0

class AudioProcessor:
    """
    Audio Processor class that leverages PyTorchaudio to provide functionalities
    for loading, cutting, and handling audio waveforms.

    Attributes:
        waveform: torch.Tensor
            The audio waveform tensor.
        sr: int
            The sample rate of the audio.
    """
    
    def __init__(self, waveform: torch.Tensor, sr : int = SAMPLE_RATE,
                 *args, **kwargs) -> None:
        
        """
        Initialize the AudioProcessor object.

        Args:
            waveform (torch.Tensor): The audio waveform tensor.
            sr (int, optional): The sample rate of the audio. Defaults to SAMPLE_RATE.
            args: Additional arguments.
            kwargs: Additional keyword arguments, e.g., device to use for processing. 
            If CUDA is available, it defaults to CUDA.

        Raises:
            ValueError: If the provided sample rate is not of type int.
        """
        
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
                
        self.waveform = waveform.to(device)
        self.sr = sr
        
        if not isinstance(self.sr, int):
            raise ValueError("Sample rate should be a single value of type int," \
                             f"not {len(self.sr)} and type {type(self.sr)}")
        
    @classmethod
    def from_file(cls, file: str, *args, **kwargs) -> 'AudioProcessor':
        """
        Create an AudioProcessor instance from an audio file.

        Args:
            file (str): The audio file path.

        Returns:
            AudioProcessor: An instance of the AudioProcessor class containing the loaded audio.
        """
        
        audio, sr = cls.load_audio(file , *args, **kwargs)

        audio = torch.from_numpy(audio)
        
        return cls(audio, sr)
    
    
    def cut(self, start: float, end: float) -> torch.Tensor:
        """
        Cut a segment from the audio waveform between the specified start and end times.

        Args:
            start (float): Start time in seconds.
            end (float): End time in seconds.

        Returns:
            torch.Tensor: The cut waveform segment.
        """
        
        start = int(start * self.sr)
        if (isinstance(end, float) or isinstance(end, int)) and isinstance(self.sr, int):
            end = int(np.ceil(end * self.sr))
        else:
            end = int(torch.ceil(end * self.sr))
        return self.waveform[start:end]

    @staticmethod
    def load_audio(file: str, sr: int = SAMPLE_RATE):
        """
        Open an audio file and read it as a mono waveform, resampling if necessary.
        This method ensures compatibility with pyannote.audio
        and requires the ffmpeg CLI in PATH.

        Args:
            file (str): The audio file to open.
            sr (int, optional): The desired sample rate. Defaults to SAMPLE_RATE.

        Returns:
            tuple: A NumPy array containing the audio waveform in float32 dtype
                    and the sample rate.

        Raises:
            RuntimeError: If failed to load audio.
        """
        # This launches a subprocess to decode audio while down-mixing
        # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
        # fmt: off
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        # fmt: on
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / NORMALIZATION_FACTOR
        
        return out , sr
    
    def __repr__(self) -> str:
        return f'TorchAudioProcessor(waveform={len(self.waveform)}, sr={int(self.sr)})'
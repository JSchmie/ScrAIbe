import os
from warnings import warn

import torch
from pydub import AudioSegment
from torchaudio import load, save


class AudioProcessor:
    def __init__(self, audio_file:str):
        
        self.audio = AudioSegment.from_file(audio_file, 
                                            format=audio_file.split('.')[-1])
        self.audio_file_path = audio_file 
        self.waveform = self.pydub_to_tensor[0]
        self.sr = self.pydub_to_tensor[1]
        
    @property
    def pydub_to_tensor(self):
        """
        Converts pydub audio segment into np.float32 of shape 
        [duration_in_seconds*sample_rate, channels],
        where each value is in range [-1.0, 1.0]. 
        Returns tuple (audio_np_array, sample_rate).
        """
        audio = self.audio
        x = torch.Tensor(audio.get_array_of_samples()
                         ).reshape((-1, audio.channels))
        y = (1 << (8 * audio.sample_width - 1))
        return x / y, audio.frame_rate
        
    def convert_audio(self, path: str, remove_orginal: bool = False, 
                      *args, **kwargs) ->  None:
        """
        Convert and saves video file or other audio files to a different file type,
        Can be used to ensure that the audio file is in the correct format
        for the Whisper model.
        :param path : path to save file
        :param remove_orginal: remove original file
        :param args: arguments for pydub.AudioSegment.export
        :param kwargs: keyword arguments for pydub.AudioSegment.export
            e.g. format
        :return: None
        """

        self.audio.export(path, *args, **kwargs)

        if remove_orginal:
            os.remove(self.audio_file_path)
            print(f'File {self.audio_file_path} removed')
        
        self.audio_file_path = path


    def to_mp3(self, *args, **kwargs) -> None:
        """
        Convert audio file to mp3 file
        :param file: audio file
        :param remove_orginal: remove original file
        :return: mp3 file path
        """
        
        warn(DeprecationWarning, "This function is deprecated," \
             "please use convert_audio instead")
        
        if "mp3" not in kwargs["format"]:
            kwargs["format"] = "mp3"
            
        self.convert_audio(*args, **kwargs)

    def to_wav(self,*args, **kwargs) -> None:
        """
        Convert audio file to wav file
        :param file: audio file
        :param remove_orginal: remove original file
        :return: wav file path
        """
        warn(DeprecationWarning, "This function is deprecated," \
             "please use convert_audio instead")
        
        if "wav" not in kwargs["format"]:
            kwargs["format"] = "wav"
            
        self.convert_audio(*args, **kwargs)

    def slower_mp3(self, path: str,
                    speed: float = 0.75,
                    type: str = "mp3") -> None:
        """
        Slow down mp3 file
        :param file: mp3 file
        :param speed: speed
        :return: None
        """

        sound = self.audio_file
        slow_sound = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * speed)
        })

        slow_sound.export(path, format=type)

        return slow_sound
    

class TorchAudioProcessor:
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
    
    
        
    @classmethod
    def from_file(cls, file: str, *args, **kwargs) -> 'TorchAudioProcessor':
        """
        Load audio file
        :param file: audio file
        :return: AudioProcessor
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f'File {file} not found')
        
        if "format" not in kwargs:
            kwargs["format"] = file.split('.')[-1]
            
        audio, sr = load(file , *args, **kwargs)
        
        return cls(audio, sr)
    
    @classmethod
    def from_ffmpeg(cls, file: str, *args, **kwargs) -> 'TorchAudioProcessor':
        """
        Initialise audio processor using pydub audio segment.
        pydub uses ffmped instead of SoX (which is used by torchaudio)
        :param file: audio file
        :return: TorchAudioProcessor
        """
        audio = AudioProcessor(file)
        
        return cls(audio.waveform, audio.sr)
        

    def cut(self, start: float, end: float) -> torch.Tensor:
        """
        Cut audio file
        :param start: start time in seconds
        :param end: end time in seconds
        :return: AudioProcessor
        """
        start = int(start / self.sr)
        end = torch.ceil(end / self.sr)
        
        return self.waveform[:, start:end]
    
    def save(self, path: str, *args, **kwargs) -> None:
        """
        Save audio file
        :param path: path to save file
        :return: None
        """
        if "format" not in kwargs:
            kwargs["format"] = path.split('.')[-1]
            
        save(path, self.waveform, self.sr, *args, **kwargs)
    
    def __repr__(self) -> str:
        return f'TorchAudioProcessor(waveform={len(self.waveform)}, sr={int(self.sr)})'
    
    def __str__(self) -> str:
        return f'TorchAudioProcessor(waveform={len(self.waveform)}, sr={int(self.sr)})'
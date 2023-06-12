from typing import Any, Union
from pydub import AudioSegment
import torch
from torchaudio import load, save
import os
from warn import warn

class AudioProcessor:
    def __init__(self, audio_file:str):
        
        self.audio_file_path = audio_file
        self.audio_file = AudioSegment.from_file(audio_file, format=audio_file.split('.')[-1])

        self.audiofilename = audio_file.split('/')[-1][:-4]
        self.coreaudiofile =  audio_file.split('/')[-1][:-4]
        self.audiofilefolder = os.path.dirname(audio_file)
        self.audio_file_type = audio_file.split('.')[-1]

    
    def save(self, path: str, remove_orginal: bool = True , *args, **kwargs) -> None:
        """
        Convert and saves video file or other audio files to a different file type,
         Can be used to ensure that the audio file is in the correct format for the Whisper model
        :param path : path to save file
        :param remove_orginal: remove original file
                :return: mp3 file path
        """
        print(f'Converting {self.audiofilename} to .{type} file')

        if savefolder == "":
            savefolder = self.audiofilefolder

        if savename == "":
            savename = self.coreaudiofile + f'.{type}'
        else:
            savename = savename + f'.{type}'

        savepath = os.path.join(savefolder, savename)

        self.audio_file.export(savepath, format=type)

        if remove_orginal:
            os.remove(self.audio_file_path)
            print(f'File {self.audio_file_path} removed')



    def to_mp3(self, savefolder: str = "", savename: str = "", remove_orginal: bool = True):
        """
        Convert audio file to mp3 file
        :param file: audio file
        :param remove_orginal: remove original file
        :return: mp3 file path
        """
        warn(DeprecationWarning, "This function is deprecated, please use convert_audio instead")
        return self.convert_audio(savefolder = savefolder,
                                   savename = savename,
                                   type="mp3",
                                   remove_orginal=remove_orginal)

    def to_wav(self, savefolder: str = "",
                savename: str = "",
                remove_orginal: bool = True):
        """
        Convert audio file to wav file
        :param file: audio file
        :param remove_orginal: remove original file
        :return: wav file path
        """
        warn(DeprecationWarning, "This function is deprecated, please use convert_audio instead")
        return self.convert_audio(savefolder = savefolder, 
                                  savename = savename,type="wav",
                                  remove_orginal=remove_orginal)

    def slower_mp3(self, savefolder: str = "",
                    speed: float = 0.75,
                    type: str = "mp3"):
        """
        Slow down mp3 file
        :param file: mp3 file
        :param speed: speed
        :return: None
        """
        if savefolder == "":
            savefolder = self.audiofilefolder
        else:
            savefolder = savefolder

        sound = self.audio_file
        slow_sound = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * speed)
        })

        speedstr = str(speed).replace('.', '')

        file_out = self.coreaudiofile + f'_{speedstr}.{type}'

        save_path = os.path.join(savefolder, file_out)

        slow_sound.export(save_path, format=type)

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
            kwargs["format"] = file.split('.')[-1]
            
        save(file, self.waveform, self.sr, *args, **kwargs)
    
    def __repr__(self) -> str:
        return f'TorchAudioProcessor(waveform={len(self.waveform)}, sr={int(self.sr)})'
    
    def __str__(self) -> str:
        return f'TorchAudioProcessor(waveform={len(self.waveform)}, sr={int(self.sr)})'
    
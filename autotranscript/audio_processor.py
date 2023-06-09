from typing import Union
from pydub import AudioSegment
import os

class AudioProcessor:
    def __init__(self, audio_file:str):
        self.audio_file_path = audio_file
        self.audio_file = AudioSegment.from_file(audio_file, format=audio_file.split('.')[-1])

        self.audiofilename = audio_file.split('/')[-1][:-4]
        self.coreaudiofile =  audio_file.split('/')[-1][:-4]
        self.audiofilefolder = os.path.dirname(audio_file)
        self.audio_file_type = audio_file.split('.')[-1]



    def convert_audio(self, savefolder: str = "", savename: str = "", type: str = "wav", remove_orginal: bool = True):
        """
        Convert video file or other audio files to mp3 file, ensures that the audio file is in the correct format for the
        Whisper model
        :param file: path to audio or video file
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

        print(f'Converted {self.audiofilename} to {type}')

        if remove_orginal:
            os.remove(self.audio_file_path)
            print(f'File {self.audio_file_path} removed')

        self.audio_file_path = savepath
        self.audio_file = AudioSegment.from_file(savepath, format=type)

        return self

    def to_mp3(self, savefolder: str = "", savename: str = "", remove_orginal: bool = True):
        """
        Convert audio file to mp3 file
        :param file: audio file
        :param remove_orginal: remove original file
        :return: mp3 file path
        """
        return self.convert_audio(savefolder = savefolder, savename = savename, type="mp3", remove_orginal=remove_orginal)

    def to_wav(self, savefolder: str = "", savename: str = "", remove_orginal: bool = True):
        """
        Convert audio file to wav file
        :param file: audio file
        :param remove_orginal: remove original file
        :return: wav file path
        """
        return self.convert_audio(savefolder = savefolder, savename = savename,type="wav", remove_orginal=remove_orginal)

    def slower_mp3(self, savefolder: str = "", savename: str = "", speed: float = 0.75, type: str = "mp3"):
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
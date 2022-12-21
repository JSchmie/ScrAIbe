
import whisper
from time import time
import os

from typing import Union
from pydub import AudioSegment

class Transcribe:
    def __init__(self, audiofile : Union[bool, str, list] = None, model : str =  "medium", language :str =  "German"):
        """
         Class to autotranscript audio and video files with the Whisper model
        :param audiofile: audio file or list of audio files
        :param model: model to use for transcription
        :param language: language of the audio file
        """

        self.audiofile = audiofile

        self.language = language

        """
        Create folder structure
        """

        self.currentpath,\
            self.audiopath,\
            self.transcriptionpath,\
            self.audiofiles = self.create_folder_structure() # create folder structure

        print("loading model")
        self.model = whisper.load_model(model)  # load model
        print("model loaded")

    def create_folder_structure(self):
        """
        Create folder structure for audio and transcription files

        :return:  currentpath, audiopath, transcriptionpath, audiofiles
        """
        currentpath = os.getcwd() # get current path

        if not os.path.exists(os.path.join(currentpath, 'audiofiles')):
            print('Creating audiofiles folder')
            os.makedirs(os.path.join(currentpath, 'audiofiles'))
        if not os.path.exists(os.path.join(currentpath, 'transcription')):
            print('Creating transcription folder')
            os.makedirs(os.path.join(currentpath, 'transcription'))

        audiopath = os.path.join(currentpath, 'audiofiles')  # path to audio files
        transcriptionpath = os.path.join(currentpath, 'transcription') # path to transcription files

        audiofiles = os.listdir(audiopath) # list of audio files

        return currentpath, audiopath, transcriptionpath, audiofiles

    def to_mp3(self,file,  remove_orginal=True):
        """
        Convert video file or other audio files to mp3 file, ensures that the audio file is in the correct format for the
        Whisper model
        :param file:  audio or video file
        :param remove_orginal: remove original file
        :return: mp3 file path
        """

        AudioSegment.from_file(file, format=file.split('.')[-1]).export(file[:-4] + '.mp3', format='mp3')

        if remove_orginal:
            os.remove(file)
            print(f'File {file} removed')
        return os.path.join(file[:-4] + '.mp3')


    def transcribe(self):

        if self.audiofile is not None:
            if self.audiofile in self.audiofiles:
                audiofile = os.path.join(self.audiopath, self.audiofile)
            else:
                raise ValueError('Audio file not found')

            if not audiofile.endswith('.mp3'):
                print('Converting video to audio')
                audiofile = self.to_mp3(audiofile)

            print(f'Start transcribing Audio file: {audiofile}')
            _stime = time()
            result = self.model.transcribe(audiofile, verbose=True, language= self.language)

            print(f'Transcription finished in {time() - _stime} seconds')

            txtfilename = str(audiofile.split('/')[-1][:-4]) + '.txt'

            savepath = os.path.join(self.transcriptionpath, txtfilename)

            with open(savepath, 'w') as f:
                f.write(result["text"])
        elif self.audiofile is None or isinstance(self.audiofile, list):
            print('No audio file specified or list of audio files')
            print(f"{len(self.audiofiles)} audio files found in {self.audiopath}")
            print("Start transcribing all audio files")
            i = 0
            for audiofile in self.audiofiles:

                audiofile = os.path.join(self.audiopath, audiofile)

                if not audiofile.endswith('.mp3'):
                    audiofile = self.to_mp3(audiofile)

                print(f'Start transcribing Audio file: {audiofile}')
                _stime = time()
                result = self.model.transcribe(audiofile, verbose=True, language=self.language)
                print(f'Transcription finished in {time() - _stime} seconds')

                txtfilename = str(audiofile.split('/')[-1][:-4]) + '.txt'

                savepath = os.path.join(self.transcriptionpath, txtfilename)

                with open(savepath, 'w') as f:
                    f.write(result["text"])

                i += 1
                print(f'{i} of {len(self.audiofiles)} files transcribed')

        else:
            raise ValueError('Audio file not found')

        print('Transcription finished')

    def __call__(self):
        return self.transcribe()
    def __repr__(self):
        return f"Transcribe(audiofile={self.audiofile}, model={self.model}, language={self.language})"
    def __str__(self):
        return f"Transcribe(audiofile={self.audiofile}, model={self.model}, language={self.language})"

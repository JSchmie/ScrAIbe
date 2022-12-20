
import whisper
from time import time
import os
from moviepy.editor import *
from typing import Union

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

    def video_to_audio(self,file,  remove_video=True):
        clip = VideoFileClip(file)
        clip.audio.write_audiofile(os.path.join(file[:-4] + '.mp3'))
        if remove_video:
            os.remove(file)
            print(f'Video {file} removed')
        return os.path.join(file[:-4] + '.mp3')


    def transcribe(self):

        if self.audiofile is not None:
            if self.audiofile in self.audiofiles:
                audiofile = os.path.join(self.audiopath, self.audiofile)
            else:
                raise ValueError('Audio file not found')

            if audiofile.endswith('.mp4'):
                print('Converting video to audio')
                audiofile = self.video_to_audio(audiofile)

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

                if audiofile.endswith('.mp4'):
                    audiofile = self.video_to_audio(audiofile)

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


import whisper
from time import time, sleep
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
    def check_if_allready_transcribed(self, filename):
        """
        Check if all audio files are already transcribed
        :param filename: audio file name
        :return: bool
        """
        purefilename = filename.split('/')[-1][:-4] + '.txt'
        if purefilename in os.listdir(self.transcriptionpath):
            print(f'File {purefilename[:-4]} already transcribed')
            return True
        else:
            return False
    def to_mp3(self,file,  remove_orginal=True):
        """
        Convert video file or other audio files to mp3 file, ensures that the audio file is in the correct format for the
        Whisper model
        :param file:  audio or video file
        :param remove_orginal: remove original file
        :return: mp3 file path
        """
        print(f'Converting {file} to mp3')
        AudioSegment.from_file(file, format=file.split('.')[-1]).export(file[:-4] + '.mp3', format='mp3')
        print(f'Converted {file} to mp3')
        if remove_orginal:
            os.remove(file)
            print(f'File {file} removed')
        return os.path.join(file[:-4] + '.mp3')

    def slower_mp3(self, file, speed=0.5):
        """
        Slow down mp3 file
        :param file: mp3 file
        :param speed: speed
        :return: None
        """
        if not os.path.exists(os.path.join(self.transcriptionpath, 'slower_version')):
            print('Creating slower_version folder')
            os.makedirs(os.path.join(self.transcriptionpath, 'slower_version'))

        path = os.path.join(self.transcriptionpath, 'slower_version')

        sound = AudioSegment.from_file(file, format="mp3")
        slow_sound = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * speed)
        })
        speedstr = str(speed).replace('.', '')
        file_out = file.split('/')[-1][:-4] + f'_{speedstr}.mp3'
        save_path = os.path.join(path, file_out)
        slow_sound.export(save_path, format="mp3")

    def transcribe(self, speed = 0.75):

        if self.audiofile is not None:
            if self.audiofile in self.audiofiles:
                audiofile = os.path.join(self.audiopath, self.audiofile)
            else:
                raise ValueError('Audio file not found')

            if not self.check_if_allready_transcribed(self.audiofile):

                if not audiofile.endswith('.mp3'):
                    print('Converting video to audio')
                    audiofile = self.to_mp3(audiofile)
                if speed != 1:
                    print('Creating slower version of the audio file with speed {}'.format(speed))
                    self.slower_mp3(audiofile, speed=speed)

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

                if not self.check_if_allready_transcribed(audiofile):

                    if not audiofile.endswith('.mp3'):
                        audiofile = self.to_mp3(audiofile)
                    if speed != 1:
                        print('Creating slower version of the audio file with speed {}'.format(speed))
                        self.slower_mp3(audiofile, speed=speed)

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


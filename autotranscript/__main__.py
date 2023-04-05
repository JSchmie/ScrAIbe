
import whisper
from time import time, sleep
import os
import glob
import re
import shutil
import sys
from tqdm import tqdm

from typing import Union
from pydub import AudioSegment

from pyannote.audio import Pipeline

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

class WhisperTranscription:
    def __init__(self, audio_file: str , model, language: str = "German"):

        self.audio_file = audio_file
        self.model = model
        self.language = language

    def transcribe(self, language:str = "German"):
        """
        Transcribe audio file

        language: language of the audio file
        :return: transcript as string
        """

        audiofilename = self.audio_file.split('/')[-1]
        #print(f'Start transcribing Audio file: {audiofilename}')

        _stime = time()
        result = self.model.transcribe(self.audio_file, language=self.language)

        #print(f'Transcription finished in {time() - _stime} seconds')

        self.transcript = result

        return result["text"]

    def save_transcript(self, transcript:str = "", savefolder : str = "", savename: str = ""):
        """
        Save transcript to file
        :param transcript: transcript as string
        :param savefolder: folder to save transcript
        :param savename: name of the transcript file
        :return: None
        """
        if savefolder == "":
            savefolder = os.path.dirname(self.audio_file)
        else:
            savefolder = savefolder

        if savename == "":
            savename = self.audio_file.split('/')[-1][:-4] + '.txt'
        else:
            savename = savename

        if transcript == "":
            transcript = self.transcript["text"]

        savepath = os.path.join(savefolder, savename)

        with open(savepath, 'w') as f:
            f.write(transcript)

        print(f'Transcript saved to {savepath}')

class Diarisation(AudioProcessor):
    def __init__(self, audio_file: str, model,**kwargs):

        super().__init__(audio_file=audio_file)

        self.model = model


    def diarization(self, *args, **kwargs):

        if "num_speakers" in kwargs:
            num_speakers = kwargs['num_speakers']
            kwargs.pop('num_speakers')
        else:
            num_speakers = 2

        audiofilename = self.coreaudiofile

        print(f'Start diarization of audio file: {self.audiofilename}')

        _stime = time()

        diarization = self.model(self.audio_file_path, num_speakers=num_speakers)

        print(f'Diarization finished in {time() - _stime} seconds')
        self.diarization = diarization

        return diarization

    def format_diarization_output(self, *args, **kwargs):
        """
        Format diarization output to a list of tuples
        :param args:
        :param kwargs:
        :return: dict with speaker names as keys and list of tuples as values and list of different speakers
        """

        diarization_output = {"speakers": [], "segments": []}

        if not hasattr(self, 'diarization'):
            # ensure diarization is run before formatting
            self.diarization = self.diarization()


        for segment, _, speaker in self.diarization.itertracks(yield_label=True):
            diarization_output["speakers"].append(speaker)
            diarization_output["segments"].append(segment)

        normalized_output = []
        index_start_speaker = 0
        index_end_speaker = 0
        current_speaker = str()

        for i, speaker in enumerate(diarization_output["speakers"]):

            if i == 0:
                current_speaker = speaker

            if speaker != current_speaker:

                index_end_speaker = i - 1

                normalized_output.append([index_start_speaker, index_end_speaker, current_speaker])

                index_start_speaker = i
                current_speaker = speaker

            if i == len(diarization_output["speakers"]) - 1:

                index_end_speaker = i
                normalized_output.append([index_start_speaker, index_end_speaker, current_speaker])


        self.normalized_output = normalized_output
        self.diarization_output = diarization_output

        return diarization_output,normalized_output

    def create_temporary_wav(self,savefolder: str = "", savename: str = "", *args, **kwargs):
        """
        Create temporary wav file for diarization
        :param savefolder: folder to save the temporary wav file
        :param savename: name of the temporary wav file prefix
        :param audiofile: audio file
        :return: temporary wav file
        """


        if savefolder == "":
            folder = '.temp'
            if not os.path.exists(folder):
                os.makedirs(folder)
        else:
            folder = savefolder

        folder = os.path.realpath(folder)

        if savename == "":
            savename = self.coreaudiofile + '.wav'
        else:
            savename = savename


        if not os.path.exists(folder):
            os.makedirs(folder)

        if not hasattr(self, 'normalized_output') or not hasattr(self, 'diarization_output'):
            self.format_diarization_output()


        speaker = set(self.diarization_output["speakers"])
        num_speak_iter = [0 for _ in range(len(speaker))]

        for count, outp in enumerate(self.normalized_output):
            start = self.diarization_output["segments"][outp[0]].start
            end = self.diarization_output["segments"][outp[1]].end

            print("start: ", start)
            print("end: ", end)

            start_milliseconds = start * 1000
            end_milliseconds = end * 1000

            print("start_milliseconds: ", start_milliseconds)
            print("end_milliseconds: ", end_milliseconds)

            print("cut audio")

            cut_audio = self.audio_file[start_milliseconds:end_milliseconds]

            print("save audio")
            print(f".temp/{count}_speaker_" + str(outp[2]) + ".wav")
            cut_audio.export(f".temp/{count}_speaker_" + str(outp[2]) + ".wav", format="wav")

        return os.path.realpath(folder)

    def __repr__(self):
        return f"Diarization(audiofile={self.audiofile}, model={self.model}, language={self.language})"
    def __str__(self):
        return f"Diarization(audiofile={self.audiofile}, model={self.model}, language={self.language})"


class AutoTranscribe:
    def __init__(self, audiofile: Union[str, bool, list] = None,
                 model: str = "medium",
                 language: str = "German",
                 diarisation: bool = False,
                 audioinput: str = "audiofiles",
                 transcriptionout: str = "transcriptions",
                 *args, **kwargs):
        """
        AutoTranscribe
        :param audiofile: audio file or list of audio files to transcribe
        :param model: model name (default: medium)
        :param language: language (default: German)
        :param diarisation: diarisation (default: False)
        """
        if audiofile is None:
            audiofile = os.listdir(audioinput) # get all audio files in audioinput folder
            audiofile = [os.path.realpath(os.path.join(audioinput, file)) for file in audiofile]# add path to audio files

        self.audiofile = audiofile
        self.language = language
        self.diarisation = diarisation
        if diarisation:
            print("Diarisation is enabled")
            print("Load Diarisation model")
            self.diarisation_model = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                             use_auth_token = self._get_token())
            print("Load Diarisation model done")

        print(f"Load Whisper model {model}")
        self.model = whisper.load_model(model)
        print(f"Load Whisper model {model} done")

        self.currentpath, \
            self.audiopath, \
            self.transcriptionpath, \
            self.audiofiles = self.create_folder_structure(audioinput, transcriptionout)  # create folder structure



    def transcribe(self, *args, **kwargs):

        if isinstance(self.audiofile, str):
            for i in range(len(self.audiofiles)):
                if self.audiofile in self.audiofiles[i]:
                    self.audiofile = [self.audiofiles[i]]
                    break

            audiolist = self.audiofile

        elif isinstance(self.audiofile, list):
            audiolist = self.audiofile
        else:
            audiolist = self.audiofiles

        if not set(audiolist).issubset(set(self.audiofiles)):
            raise ValueError(f"Audio file {self.audiofile} not found in {self.audiopath}")


        for audiofile in audiolist:
            _start = time()
            if not "/" in audiofile:
                audiofile = os.path.join(self.audiopath, audiofile)

            if not self.check_if_already_transcribed (audiofile):

                audio = AudioProcessor(audiofile)

                if not audiofile.endswith('wav'):
                    audio = audio.to_wav()
                    self.audiofile = audio.audio_file_path
                    audiofile = audio.audio_file_path

                if "speed" in kwargs:
                    speed = kwargs['speed']
                    kwargs.pop('speed')

                    print('Creating slower version of the audio file with speed {}'.format(speed))
                    slower_audio = os.path.join(self.transcriptionpath, 'slower_version')
                    if not os.path.exists(slower_audio):
                        os.makedirs(slower_audio)
                    audio.slower_mp3(savefolder=slower_audio,speed=speed)

                if not self.diarisation:
                    WhisperTranscription(audiofile, self.model, self.language
                                         ).save_transcript(savefolder = self.transcriptionpath)

                else:
                    print("Start diarisation")
                    dia = Diarisation(audiofile, self.diarisation_model)

                    if 'num_speakers' in kwargs:
                        num_speakers = kwargs['num_speakers']
                        kwargs.pop('num_speakers')
                        dia.diarization(num_speakers=num_speakers)
                    else:
                        dia.diarization()

                    temppath = dia.create_temporary_wav()
                    temppath_dict, _ = dia.format_diarization_output()
                    speakers = list(set(temppath_dict["speakers"]))


                    fstring = "\\begin{drama}"

                    for speaker in speakers:
                        speaker = speaker.replace("SPEAKER_", "")
                        fstring += "\n\t\Character{S"+ str(speaker) + "}{S" + str(speaker) + "}"


                    files = glob.glob(temppath + "/*.wav")

                    # Sort files according to the digits included in the filename
                    files = sorted(files, key=lambda x: float(re.findall("(\d+)", x)[0]))

                    for file in tqdm(files):

                            Whisper = WhisperTranscription(file, self.model, self.language).transcribe()

                            for s in speakers:
                                if s in file:
                                    s = s.replace("SPEAKER_", "")
                                    fstring += f"\n\S{s}speaks: \n {Whisper}"

                    fstring += "\n\end{drama}"

                    print(fstring)

                    with open(os.path.join(self.transcriptionpath,
                                           os.path.basename(audiofile).split('.')[0] + '.tex'), 'w') as f:
                        f.write(fstring)

                    print("Remove temporary files")
                    shutil.rmtree(temppath)

                print(f"Transcription of {audiofile} done in total of {time() - _start} seconds")

    def create_folder_structure(self, audiopath: str, transcriptionout: str):
        """
        Create folder structure for audio and transcription files

        :return:  currentpath, audiopath, transcriptionpath, audiofiles
        """
        currentpath = os.path.dirname(sys.argv[0]) # get executable path

        if not os.path.exists(os.path.join(currentpath, audiopath)):
            print('Creating audiofiles folder')
            os.makedirs(os.path.join(currentpath, audiopath))
        if not os.path.exists(os.path.join(currentpath, transcriptionout)):
            print('Creating transcription folder')
            os.makedirs(os.path.join(currentpath, transcriptionout))

        audiopath = os.path.join(currentpath, audiopath)  # path to audio files
        transcriptionpath = os.path.join(currentpath, transcriptionout)  # path to transcription files


        _audiofiles =  os.listdir(audiopath) # list of audio files
        audiofiles = []
        for i in _audiofiles:
                audiofiles.append(os.path.join(audiopath, i))

        return currentpath, audiopath, transcriptionpath, audiofiles

    def check_if_already_transcribed (self, filename: str):
        """
        Check if all audio files are already transcribed
        :param filename: audio file name
        :return: bool
        """
        purefilename = filename.split('/')[-1][:-4]
        _files = os.listdir(self.transcriptionpath)
        for i,f in enumerate(_files):
            _files[i] = f[:-4]

        if purefilename in _files:
            print(f'File {purefilename[:-4]} already transcribed')
            return True
        else:
            return False
    @classmethod
    def _get_token(self):
        # check ig .pyannotetoken.txt exists
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.pyannotetoken')
        if os.path.exists(path):
            with open(path, 'r') as f:
                token = f.read()
        else:
            raise ValueError('No token found. Please create a token at https://huggingface.co/settings/token'
                             ' and save it in a file called .pyannotetoken.txt')
        return token

    def __repr__(self):
        return f"AutoTranscribe(audiofile={self.audiofile}, model={self.model}, language={self.language}, diarisation={self.diarisation})"
    def __call__(self, *args, **kwargs):
        return self.transcribe(*args, **kwargs)

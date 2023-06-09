from audio_processor import AudioProcessor
from time import time
import os

class Diarisation(AudioProcessor):
    def __init__(self, audio_file: str, model,**kwargs) -> None:

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
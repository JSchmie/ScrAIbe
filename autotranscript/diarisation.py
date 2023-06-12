from pyannote.audio import Pipeline
from time import time
import os
from typing import TypeVar

Annotation = TypeVar('Annotation') 

PYANNOTE_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "models", "pyannote", 
                                     "speaker_diarization", "config.yaml")

class Diarisation:
    def __init__(self, model,*args,**kwargs) -> None:

        self.model = model


    def diarization(self, audiofile : str , *args, **kwargs) -> Annotation:
        """
        Diarization of audio file
        :param audiofile: path to audio file
        :param args: args for diarization model 
        :param kwargs: kwargs for diarization model
        :return: diarization
        """

        print(f'Start diarization of audio file: {audiofile}')

        diarization = self.model(audiofile,*args, **kwargs)

        print('Diarization finished')

        out = self.format_diarization_output(diarization)

        return out

    @staticmethod
    def format_diarization_output(dia : Annotation) -> dict:
        """
        Format diarization output to a list of tuples
        :param dia: diarization output
        :return: dict with speaker names as keys and list of tuples
                 as values and list of different speakers
        """

        dia_list  = list(dia.itertracks(yield_label=True))
        diarization_output = {"speakers": [], "segments": []}

        normalized_output = []
        index_start_speaker = 0
        index_end_speaker = 0
        current_speaker = str()
        
        ###
        # Sometimes two consecutive speakers are the same
        # This loop removes these duplicates
        ###


        for i, (_, _, speaker) in enumerate(dia_list):
            
            if i == 0:
                current_speaker = speaker

            if speaker != current_speaker:

                index_end_speaker = i - 1

                normalized_output.append([index_start_speaker,
                                           index_end_speaker,
                                           current_speaker])

                index_start_speaker = i
                current_speaker = speaker

            if i == len(diarization_output["speakers"]) - 1:

                index_end_speaker = i
                normalized_output.append([index_start_speaker, 
                                          index_end_speaker, 
                                          current_speaker])
       
        for outp in normalized_output:
            #convert in milliseconds
            start =  dia_list[outp[0]][0].start * 1000
            end =  dia_list[outp[1]][0].end * 1000

            diarization_output["segments"].append([start, end])
            diarization_output["speakers"].append(outp[2])

        return diarization_output
    
    @classmethod
    def load_model(cls, model: str = PYANNOTE_DEFAULT_PATH, 
                        token: str = "",
                        local : bool = True,
                        *args, **kwargs) -> Pipeline:
        """
        Load modules from pyannote

        Parameters
        ----------
        model : str
            pyannote model 
            default: /models/pyannote/speaker_diarization/config.yaml
        token : str
            HUGGINGFACE_TOKEN
        local : bool
            If true, load from local cache
        
        Returns
        -------
        Pipeline Object
        """

        if local:
            diarization_model =  Pipeline.from_pretrained(model,*args, **kwargs)
        else:
            diarization_model =  Pipeline.from_pretrained(model, use_auth_token = token,
                                                           *args, **kwargs)
        
        return cls(diarization_model)

    def __repr__(self):
        return f"Diarisation(model={self.model})"
    def __str__(self):
        return f"Diarisation(model={self.model})"


if __name__ == '__main__':

    model = Diarisation.load_model()
    print(model)
    audiofile = "/home/jacob/PycharmProjects/autotranscript/tests/test.wav"
    out = model.diarization(audiofile)
    print(out)

    # # deprecated
    # def create_temporary_wav(self, location_of_temp_folder : str = '.temp'):
    #     """
    #     Create temporary wav file for diarization
    #     :param location_of_temp_folder: folder to save the temporary wav file
    #         default: .temp
    #     :param savename: name of the temporary wav file prefix
    #     :param audiofile: audio file
    #     :return: temporary wav file
    #     """
    #     print("Linne 84 Diarisation.py create_temporary_wav :" /
    #            "location_of_temp_folder.split('/')[-1]",location_of_temp_folder.split('/')[-1])
        
    #     if location_of_temp_folder.split('/')[-1] != '.temp':
    #         folder =os.path.join(location_of_temp_folder, '.temp')
    #     else:
    #         folder = location_of_temp_folder
        
    #     if not os.path.exists(folder):
    #             os.makedirs(folder)
        
    #     folder = os.path.realpath(folder)

    #     if not hasattr(self, 'normalized_output') or not hasattr(self, 'diarization_output'):
    #         raise AttributeError("You need to run the diarization first")
        
    #     speaker = set(self.diarization_output["speakers"])
    #     num_speak_iter = [0 for _ in range(len(speaker))]

    #     for count, outp in enumerate(self.normalized_output):
    #         print(outp)
    #         print(self.diarization_output["segments"][outp[0]])
    #         print(self.diarization_output["segments"][outp[1]])

    #         start = self.diarization_output["segments"][outp[0]].start
    #         end = self.diarization_output["segments"][outp[1]].end

    #         print("start: ", start)
    #         print("end: ", end)

    #         start_milliseconds = start * 1000
    #         end_milliseconds = end * 1000

    #         print("start_milliseconds: ", start_milliseconds)
    #         print("end_milliseconds: ", end_milliseconds)

    #         print("cut audio")

    #         cut_audio = self.audio_file[start_milliseconds:end_milliseconds]

    #         print("save audio")
    #         print(f".temp/{count}_speaker_" + str(outp[2]) + ".wav")
    #         cut_audio.export(f".temp/{count}_speaker_" + str(outp[2]) + ".wav", format="wav")

    #     return os.path.realpath(folder)
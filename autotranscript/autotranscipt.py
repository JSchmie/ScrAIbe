from audio import AudioProcessor
from diarisation import Diariser
from transcriber import Transcriber, whisper
from transcript_exporter import Transcript
from typing import Union , TypeVar
from tqdm import trange
import torch

diarisation = TypeVar('diarisation')


class AutoTranscribe:
    def __init__(self,
                whisper_model: Union[bool, str, whisper] = None,
                dia_model : Union[bool, str, diarisation] = None,
                dia_kwargs : dict = {},
                whisper_kwargs : dict = {}) -> None:
        """
        AutoTranscribe class
        
        This class is the core Api Class of the autotranscript package.
        It allows to transcribe audio files with a whisper model and
        pyannote diarization model. 
        
        Therefore it is do a fully automatic transcription of audio files.
        
        :param whisper_model: path to whisper model or whisper model
        :param dia_model: path to pyannote diarization model
        :param dia_kwargs: kwargs for pyannote diarization model
        :param whisper_kwargs: kwargs for whisper model      
        
        """
        
        if whisper_model is None:
            self.transcriber = Transcriber.load_model("medium", local=True)
            
        elif isinstance(whisper_model, str):
            self.transcriber = Transcriber.load_model(whisper_model, **whisper_kwargs)
        else:
            self.transcriber = whisper_model

        if dia_model is None:
            self.diariser = Diariser.load_model()
        elif isinstance(dia_model, str):
            self.diariser = Diariser.load_model(dia_model, **dia_kwargs)
        else:
            self.diariser = dia_model

        print("AutoTranscribe initialized all models successfully loaded.")
            
    def transcribe(self, audiofile : Union[str, torch.Tensor],
                   *args, **kwargs) -> Transcript:
        """
        Transcribe audiofile with whisper model and pyannote diarization model
        
        :param audiofile: path to audiofile or torch.Tensor
        :return: Transcript object which contains the transcript and can be used to 
                export the transcript to differnt formats.
        """
        
        audiofile = self.get_audiofile(audiofile)
        
        final_transcript = dict()
        
        dia_audio = {"waveform" : 
                        audiofile.waveform.reshape(1,len(audiofile.waveform)), 
                    "sample_rate": audiofile.sr}
       
        print("Starting diarisation.")
        
        diarisation = self.diariser.diarization(dia_audio,
                                                *args , **kwargs)
        
        print("Diarisation finished. Starting transcription.")
        
        audiofile.sr = torch.Tensor([audiofile.sr]).to(audiofile.waveform.device)
        
        for i in trange(len(diarisation["segments"]), desc= "Transcribing"):
            
            seg = diarisation["segments"][i]
            
            audio = audiofile.cut(seg[0], seg[1])
            
            transcript = self.transcriber.transcribe(audio, *args , **kwargs)
            
            final_transcript[i] = {"speaker" : diarisation["speakers"][i],
                                   "text" : transcript}

        return Transcript(transcript, diarisation)
    
    @staticmethod
    def get_audiofile(audiofile : Union[str, torch.Tensor],
                        *args, **kwargs) -> AudioProcessor:
        """
        Get audiofile as TorchAudioProcessor

        :param audiofile: path to audiofile or torch.Tensor
            :type audiofile: Union[str, torch.Tensor]
        :return: object of audiofile containes
                 waveform and sample_rate in torch.Tensor format.
            :rtype: TorchAudioProcessor
        """
        
        if isinstance(audiofile, str):
            audiofile = AudioProcessor.from_file(audiofile)   
        
        if isinstance(audiofile, torch.Tensor):
            audiofile = AudioProcessor(audiofile[0], audiofile[1])
        
        if not isinstance(audiofile, AudioProcessor):
            raise ValueError(f'Audiofile must be of type AudioProcessor,' \
                             f'not {type(audiofile)}')     
        return audiofile
    

if __name__ == "__main__":
    
    AudioTranscriber = AutoTranscribe()
    AudioTranscriber.transcribe("tests/test.wav")
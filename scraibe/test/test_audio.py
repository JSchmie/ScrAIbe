import pytest
#from scraibe import Transcriber
#from unittest.mock import patch, mock_open
#import unittest
#import os
from .audio import AudioProcessor
import torch





test_waveform = torch.tensor([]).to('cuda')
test_sr = 16000
SAMPLE_RATE = 16000
NORMALIZATION_FACTOR = 32768


@pytest.fixture
def probe_audio_processor():
    return AudioProcessor(test_waveform, test_sr)






def test_AudioProcessor_init(probe_audio_processor):
    assert isinstance(probe_audio_processor, AudioProcessor)
    assert probe_audio_processor.waveform.device == test_waveform.device
    assert torch.equal(probe_audio_processor.waveform, test_waveform)
    assert probe_audio_processor.sr == test_sr



def test_cut():
    waveform = torch.Tensor(10, 3)
    sr = 16000
    start = 4
    end = 7
    assert AudioProcessor(waveform, sr).cut(start, end).size() == int((end - start) * test_sr)



""" def test_cut(probe_audio_processor):
    start = 10
    end = 100
    test_segment =  probe_audio_processor.cut(start, end) 
    print(test_segment)
    erwartetes_segment = int((end - start) * test_sr)
    print(test_segment.size())
    assert len(test_segment) == erwartetes_segment
 """





def test_audio_processor_invalid_sr():
    with pytest.raises(ValueError):
        AudioProcessor(test_waveform, [44100,48000])


def test_audio_processor_SAMPLE_RATE():
    probe_audio_processor = AudioProcessor(test_waveform)
    assert probe_audio_processor.sr == SAMPLE_RATE       











  


    



import pytest
import os
from unittest import mock
from scraibe import Diariser



@pytest.fixture
def diariser_instance():
    with mock.patch.object(Diariser, '_get_token', return_value = 'personal Hugging-Face token')
        return Diariser('pyannote')



def test_Diariser_init(diariser_instance):
    assert diariser_instance.model == 'pyannote'



def test_diarisation_function(diariser_instance):
    with mock.patch.object(diariser_instance.model, 'apply', return_value='diarization_result'):
        diarization_output = diariser_instance.diarization('example_audio_file.wav')
        assert diarization_output == 'diarization_result'






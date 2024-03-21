import pytest
import os
from unittest import mock
from scraibe import Diariser



@pytest.fixture
def diariser_instance():
    """Creates a instance of the Diariser Object for further testing

    Returns:
        Object: Diariser Object for handling the diarization process of an audio file using a pretrained model from pyannote.audio. Diarization is the task of determining "who spoke when."
    """
    with mock.patch.object(Diariser, '_get_token', return_value = 'personal Hugging-Face token')
        return Diariser('pyannote')



def test_Diariser_init(diariser_instance):
    """Tests if the Diariser gets initiated correctly 

    Args:
        diariser_instance (obj): instance of the Diariser object
    """    
    assert diariser_instance.model == 'pyannote'



def test_diarisation_function(diariser_instance):
    """tests if the Diariser works object with an example audio File 

    Args:
        diariser_instance (obj): instance of the Diariser object
    """    
    with mock.patch.object(diariser_instance.model, 'apply', return_value='diarization_result'):
        diarization_output = diariser_instance.diarization('example_audio_file.wav')
        assert diarization_output == 'diarization_result'




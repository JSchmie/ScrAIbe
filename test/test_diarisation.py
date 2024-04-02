import pytest
import os
from unittest import mock
from scraibe import diarisation, Diariser



@pytest.fixture
def diariser_instance():
    """Fixture for creating an instance of the Diariser class with mocked token.

    This fixture is used to create an instance of the the Diariser class with a mocked token returned by the _get_token method. It patches the _get_token method of the Diariser class
    using unit.test.mock.patch.object, ensuring that it returns a predetrmined value ('personal Hugging-Face token'). The mocked Diariser object is retunrned and can be used as a dependency in otehr tests.

    Returns:
        Diariser(Obj): An instance of the Diariser class with a mocked token.
    """
    with mock.patch.object(Diariser, '_get_token', return_value = 'HF_TOKEN' ):
        return Diariser('pyannote')



def test_Diariser_init(diariser_instance):
    """Test the initialization of the Diariser class.

    This test verifies that the Diariser class is correctly initialized with the specified model.
    It checks whether the 'model' attribute of the instantiated Diariser object equals 'pyannote'.


    Args:
        diariser_instance (obj): instance of the Diariser class

    Returns: 
           None
    """    
    assert diariser_instance.model == 'pyannote'



def test_diarisation_function(diariser_instance):
    """Test the diarization function of the Diariser class.

    This test verifies that the diarization function of the Diariser class correctly processes
    an audio file and returns the diarization result. It patches the apply method of the model
    attribute of the Diariser instance using unittest.mock.patch.object, ensuring that it returns 
    a predetermined value ('diarization_result') when called with the audio file argument.
    It then calls the diarization function with an example audio file and checks whether the returned 
    diarization output matches the expected result ('diarization_result').

    Args:
        diariser_instance (obj): instance of the Diariser object

    Returns:
        None    
    """    
    with mock.patch.object(diariser_instance.model, 'apply', return_value='diarization_result'):
        diarization_output = diariser_instance.diarization('example_audio_file.wav')
        assert diarization_output == 'diarization_result'




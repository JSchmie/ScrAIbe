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
    #with mock.patch.object(Diariser, '_get_token', return_value = 'HF_TOKEN' ):
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












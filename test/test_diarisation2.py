import pytest
import os
import unittest
from unittest import mock  
from scraibe import Diariser
import torch





@pytest.fixture
def diariser_instance():
     return Diariser('pyannote')



def test_diariser_init(diariser_instance):
    assert diariser_instance.model == 'pyannote'



""" def test_format_diarization_output():
    dialogue = [("speaker1", "segment1"),("speaker2", "segment2"), ("speaker1","segment3")] 
    formatted_output = Diariser.format_diarization_output(dialogue)
    assert formatted_output == {"speakers": ["speaker1", "speaker2", "speaker1"], "segments": ["segment1", "segment2", "segment3"]} """

    
def test_get_diarisation_kwargs():
    kwargs = {"arg1": 1, "arg3": 3} 
    valid_kwargs = Diariser._get_diarisation_kwargs(**kwargs)
    assert not valid_kwargs == {"arg1": 1, "arg3": 3}   






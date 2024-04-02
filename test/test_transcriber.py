import pytest
from unittest.mock import patch
from scraibe import Transcriber



""" 
@pytest.mark.parametrize("audio_file, expected_transcription",[("path_to_test_audiofile", "test_transcription")] )
@patch("scraibe.Transcriber.load_model")

def test_transcriber(mock_load_model, audio_file, expected_transcription):
    

    Args:
        mock_load_model (_type_): _description_
        audio_file (_type_): _description_
        expected_transcription (_type_): _description_
       
    mock_model = mock_load_model.return_value
    mock_model.transcribe.return_value ={"text": expected_transcription} 

    transcriber = Transcriber.load_model(model="medium")

    transcription_result = transcriber.transcribe(audio=audio_file)

    assert transcription_result == expected_transcription """




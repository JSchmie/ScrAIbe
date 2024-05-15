import pytest
from scraibe import Transcriber
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_WAVEFORM = "Hello World"

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


@pytest.fixture
def transcriber_instance():
    return Transcriber.load_model('medium')


def test_transcriber_initialization(transcriber_instance):
    assert isinstance(transcriber_instance, Transcriber)


def test_get_whisper_kwargs():
    kwargs = {"arg1": 1, "arg3": 3}
    valid_kwargs = Transcriber._get_whisper_kwargs(**kwargs)
    assert not valid_kwargs == {"arg1": 1, "arg3": 3}


def test_transcribe(transcriber_instance):
    model = transcriber_instance
    # mocker.patch.object(transcriber_instance.model, 'transcribe', return_value={'Hello, World !'} )
    transcript = model.transcribe('test/audio_test_2.mp4')
    assert isinstance(transcript, str)

import pytest
from scraibe import Scraibe, Diariser, Transcriber, Transcript
import os


@pytest.fixture
def create_scraibe_instance():
    if "HF_TOKEN" in os.environ:
        return Scraibe(use_auth_token=os.environ["HF_TOKEN"], whisper_model= "tiny")
    else:
        return Scraibe()


def test_scraibe_init(create_scraibe_instance):
    model = create_scraibe_instance
    assert isinstance(model.transcriber, Transcriber)
    assert isinstance(model.diariser, Diariser)


def test_scraibe_autotranscribe(create_scraibe_instance):
    model = create_scraibe_instance
    transcript = model.autotranscribe('./test/audio_test_2.mp4')
    assert isinstance(transcript, Transcript)


def test_scraibe_diarization(create_scraibe_instance):
    model = create_scraibe_instance
    diarisation_result = model.diarization('./test/audio_test_2.mp4')
    assert isinstance(diarisation_result, dict)


def test_scraibe_transcribe(create_scraibe_instance):
    model = create_scraibe_instance
    transcription_result = model.transcribe('./test/audio_test_2.mp4')
    assert isinstance(transcription_result, str)


""" def test_remove_audio_file(create_scraibe_instance):
    model = create_scraibe_instance
    with pytest.raises(ValueError):
        model.remove_audio_file("non_existing_audio_file")

    model.remove_audio_file("audio_test_2.mp4")
    assert not os.path.exists("audio_test_2.mp4")   """


""" def test_get_audio_file(create_scraibe_instance):
    model = create_scraibe_instance
    audio_file = os.path.exist("audio_test_2.mp4")
    assert isinstance(audio_file, AudioProcessor)
    assert isinstance(audio_file.waveform, torch.Tensor)
    assert isinstance(audio_file.sr, torch.Tensor)   """

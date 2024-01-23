import pytest
from .audio import AudioProcessor
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test_waveform = torch.tensor([]).to(device)
TEST_SR = 16000
SAMPLE_RATE = 16000
NORMALIZATION_FACTOR = 32768


@pytest.fixture
def probe_audio_processor():
    """Creates a dummy AudioProcessor Object

    Returns:
        AudioProcessor Object with given parameters test_waveform and TEST_SR
    """    
    return AudioProcessor(test_waveform, TEST_SR)






def test_AudioProcessor_init(probe_audio_processor):
    """
    testing if the audio_processor Object gets initialized correctly

    Args: probe_audio_processor Object
        
    """    
    assert isinstance(probe_audio_processor, AudioProcessor)
    assert probe_audio_processor.waveform.device == test_waveform.device
    assert torch.equal(probe_audio_processor.waveform, test_waveform)
    assert probe_audio_processor.sr == test_sr



def test_cut():
    """Test for the test_cut Method for fixed parameters
    """    
    waveform = torch.Tensor(10, 3)
    sr = 16000
    start = 4
    end = 7
    assert AudioProcessor(waveform, sr).cut(start, end).size() == int((end - start) * TEST_SR)




   





def test_audio_processor_invalid_sr():
    """Testing the audio_processor Object with invalid Sample rate
    """    
    with pytest.raises(ValueError):
        AudioProcessor(test_waveform, [44100,48000])


def test_audio_processor_SAMPLE_RATE():
    """Making sure Sample Rate of Audio_processor Sample Rate matches global Sample Rate
    """    
    probe_audio_processor = AudioProcessor(test_waveform)
    assert probe_audio_processor.sr == SAMPLE_RATE       











  


    



import pytest
from ..scraibe.audio import AudioProcessor
import torch



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_WAVEFORM = torch.tensor([]).to(DEVICE)
TEST_SR = 16000
SAMPLE_RATE = 16000
NORMALIZATION_FACTOR = 32768


@pytest.fixture
def probe_audio_processor():
    """Fixture for creating an instance of the AudioProcessor class with test waveform and sample rate. 
    
    This fixture is used to create an instance of the AudioProcessor class with a predfined test waveform and sample rate (TEST_SR). It returns the instantiated AudioProcessor , which can bes used as a 
    dependency in other test functions.


    Returns:
        AudioProcessor (obj): An instance of the AudioProcessor class with the test waveform and sample rate.
    """    
    return AudioProcessor(TEST_WAVEFORM, TEST_SR)






def test_AudioProcessor_init(probe_audio_processor):
    """
    Test the initialization of the AudioProcessor class.

    This test verifies that the AUdioProcessor class is correctly initialized with the provided waveform and sample rate. It checks whether the instantiated AhdioProcessor object has the correct attributes
    and whether the waveform and sample rate match the expected values.

    Args:
        probe_audio_processor (obj): An instance of the AudioProcessor class to be tested.


    Returns:
           None

    

        
    """    
    assert isinstance(probe_audio_processor, AudioProcessor)
    assert probe_audio_processor.waveform.device == TEST_WAVEFORM.device
    assert torch.equal(probe_audio_processor.waveform, TEST_WAVEFORM)
    assert probe_audio_processor.sr == TEST_SR



def test_cut():
    """Test the cut function of the AudioProcessor class.
    
    This test verifies that the cut function correctly extracts a segment of audio data from
     the waveform, given start and end indices. It checks whether the size of the extracted segment matches
     the expected size based on the provided start and end indices and the sample rate.

     Returns:
            None


    """    
  
    start = 4
    end = 7
    assert AudioProcessor(TEST_WAVEFORM, TEST_SR).cut(start, end).size() == int((end - start) * TEST_SR)




   





def test_audio_processor_invalid_sr():
    """Test the behavior of AudioProcessor when an invalid smaple rate is provided.
    
    This test verifies that the AudioProcessor constructor raises a ValueError when an invalid sample rate is provided. It uses the pytest.raises context manager to check if the ValueError is raised when initializing an 
    AudioProcessor object with an invalid sample rate.

    Returns:
           None
    """    
    with pytest.raises(ValueError):
        AudioProcessor(TEST_WAVEFORM, [44100,48000])


def test_audio_processor_SAMPLE_RATE():
    """Test the default sample rate of the AudioProcessor class.
    
    This test verifies that the default sample rate of the AudioProcessor class matches the expected value defined by the constant SAMPLE_RATE. It instantiates an AudioProcessor object with a test waveform
    and checks whether the sample rate attribute (sr) of the AudioProcessor object equals the predefined constant SAMPLE_RATE.

    Returns:
           None
    """    
    probe_audio_processor = AudioProcessor(TEST_WAVEFORM)
    assert probe_audio_processor.sr == SAMPLE_RATE       











  


    



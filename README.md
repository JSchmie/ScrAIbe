
# `AutoTranscript`: Fully Automated Transcription using AI 

`AutoTranscript` is a [PyTorch](https://pytorch.org/) based interface for. To enable fully auomated Transcription using AI models containing speaker diarization models:

- [whisper](https://github.com/openai/whisper): an a general-purpose speech recognition model
- [payannote-audio](https://github.com/pyannote/pyannote-audio) an open-source toolkit for speaker diarization

Therefore `AutoTranscript` can be used as a Commandline Interface a Webserver or as a Python API.

## Setup: 
For this Project, Python 3.9 were [PyTorch](https://pytorch.org/) version 1.11.0 

The following command will pull and install the latest commit from this repository, along with its Python dependencies.

    pip install https://github.com/JSchmie/autotranscript.git
  
## Example Python usage

```python
from autotranscript import AutoTranscribe

model = AutoTranscribe()

text = model.transcribe("audio.wav")

print(f"Transcription: \n{text}")

```

## Command-line usage

If you not want to control the optimization using python, you also can use the Command-line:

	autotranscript audio.wav

Run the following to view all available options:
		
	autotranscript -h


## License 

## Citation




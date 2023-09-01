
# `AutoTranscript`: Fully Automated Transcription using AI 

`AutoTranscript` is a [PyTorch](https://pytorch.org/) based interface speech-to-text tool to generate fully automated transcriptions. AutoTranscript uses AI models containing speaker diarization models:

- [whisper](https://github.com/openai/whisper): A general-purpose speech recognition model.
- [payannote-audio](https://github.com/pyannote/pyannote-audio): An open-source toolkit for speaker diarization-.

`AutoTranscript` can be used as a command-line interface, a webserver, or as a Python API.

## Install `AutoTranscript` : 

The following command will pull and install the latest commit from this repository, along with its Python dependencies.

    pip install https://github.com/JSchmie/autotranscript.git

- **Python version**: Python 3.9
- **PyTorch version**: Python 1.11.0
  
## Usage examples

### Python usage

```python
from autotranscript import AutoTranscribe

model = AutoTranscribe()

text = model.transcribe("audio.wav")

print(f"Transcription: \n{text}")

```

### Command-line usage

If you do not want to control the optimization using Python, you also can use the command-line:

	autotranscript audio.wav

Run the following to view all available options:
		
	autotranscript -h

### Documentation usage

To access the documentation run the following command from the docs/_build/html directory:

	python -m http.server

## Roadmap

- Model quantization
- Model fine-tuning
- Implementation of LLMs
- Executable for Windows

## Contact

For queries contact Jacob Schmieder at Jacob.Schmieder@dbfz.de

## License 

## Acknowledgments

Special thanks go to the colleagues of the KIDA project - especially the teams in I5 and I2 - and the BMEL (Bundesministerium für Ernährung und Landwirtschaft).



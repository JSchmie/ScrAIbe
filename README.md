
# `ScrAIbe: Streamlined Conversation Recording with Automated Intelligence Based Environment`


`ScrAIbe` is a [PyTorch](https://pytorch.org/) based interface speech-to-text tool to generate fully automated transcriptions. AutoTranscript uses AI models containing speaker diarization models:

- [whisper](https://github.com/openai/whisper): A general-purpose speech recognition model.
- [payannote-audio](https://github.com/pyannote/pyannote-audio): An open-source toolkit for speaker diarization-.

## Install `ScrAIbe` : 

The following command will pull and install the latest commit from this repository, along with its Python dependencies.

    pip install git+https://github.com/JSchmie/autotranscript.git

- **Python version**: Python 3.9
- **PyTorch version**: Python 1.11.0
  
## Usage 

`AutoTranscript` can be used as a command-line interface, a webserver, or as a Python API.

### Python usage

```python
from autotranscript import AutoTranscribe

model = AutoTranscribe()

text = model.transcribe("audio.wav")

print(f"Transcription: \n{text}")

```

Refer to [whisper](https://github.com/openai/whisper) and [payannote-audio](https://github.com/pyannote/pyannote-audio) for further options.

### Command-line usage


You can also run ScrAIbe in a [Gradio App](https://github.com/gradio-app/gradio)  interface using the following command-line:

	autotranscript audio.wav

Some example of important functionalities are:

-  `--task`: Task to be performed, either transcription, diarization or translation into English. Default is transcription.
- `--hf-token`: To download the models, a Hugging Face token must be generated. Check [Hugging Face](https://huggingface.co/docs/hub/security-tokens) for further information on how to do that.
- `--server-name`: Name of the Web Server. If empty 127.0.0.1 or 0.0.0.0 will be used
- `--whisper-model-name`: Name of the [whisper](https://github.com/openai/whisper) model to be used. Default is `medium`.


Run the following to view all available options:
		
	autotranscript -h

## Documentation 

For further insights check the [documentation page](https://cristinaortizcruz.github.io/Test/).

## Contributions

We are happy for any interest in contributing: In order to do that, fork the repo and use merge requests to incorporate your contribution.

## Roadmap

The following milestones are planned for the further development of ScrAIbe:

- Model quantization   
Quantization to empower memory and computational efficiency.

- Model fine-tuning  
In order to be able to cover a variety of linguistic phenomena.

For example, currently ScrAIbe is able to transcribe word by word, but ignores filler words or speech pauses. 
These phenomena can be addressed by fine-tuning with the corresponding data.

- Implementation of LLMs   
One example is the implementation of a summarization or extraction model, which enables ScrAIbe to automatically summarize or retrieve the key information out of a generated transcription, which could be the minutes of a meeting.

- Executable for Windows

## Contact

For queries contact [Jacob Schmieder](Jacob.Schmieder@dbfz.de)

## License 

<!-- licensing  missing? Apache 2.0 -->
ScrAIbe is licensed under (tbd).

## Acknowledgments

Special thanks go to the KIDA project and the BMEL (Bundesministerium für Ernährung und Landwirtschaft), especially to the AI Consultancy Team and the Infrastructure Team.

![KIDA](kida_dark.png#gh-dark-mode-only)    &nbsp;    ![BMEL](BMEL_dark.png#gh-dark-mode-only) &nbsp;&nbsp;&nbsp;&nbsp; ![DBFZ](DBFZ_dark.png#gh-dark-mode-only)   &nbsp;  &nbsp;&nbsp;&nbsp;    ![MRI](MRI.png#gh-dark-mode-only)   

![KIDA](kida.png#gh-light-mode-only)    &nbsp;    ![BMEL](BMEL.jpg#gh-light-mode-only) &nbsp;&nbsp;&nbsp;&nbsp; ![DBFZ](DBFZ.png#gh-light-mode-only)   &nbsp;  &nbsp;&nbsp;&nbsp;    ![MRI](MRI.png#gh-light-mode-only)  

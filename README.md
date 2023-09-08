
# `ScrAIbe: Streamlined Conversation Recording with Automated Intelligence Based Environment`

`ScrAIbe` is a state-of-the-art,  [PyTorch](https://pytorch.org/) based multilingual speech-to-text framework to generate fully automated transcriptions. 

Beyond transcription, ScrAIbe supports advanced functions, such as speaker diarization and speaker recognition.

Designed as a comprehensive AI toolkit, it uses multiple AI models:

- [whisper](https://github.com/openai/whisper): A general-purpose speech recognition model.
- [payannote-audio](https://github.com/pyannote/pyannote-audio): An open-source toolkit for speaker diarization.

The framework utilizes a PyanNet-inspired pipeline with the `Pyannote` library for speaker diarization and `VoxCeleb` for speaker embedding.

During post-diarization, each audio segment is processed by the OpenAI `Whisper` model, in a transformer encoder-decoder structure. Initially, a CNN mitigates noise and enhances speech. Before transcription, `VoxLingua` dentifies the language segment, facilitating Whisper's role in both transcription and text translation.

The following graphic illustates the whole pipeline:


## Install `ScrAIbe` : 

The following command will pull and install the latest commit from this repository, along with its Python dependencies.

    pip install git+https://github.com/JSchmie/autotranscript.git

- **Python version**: Python 3.9
- **PyTorch version**: Python 1.11.0
  
## Usage 

We've developed ScrAIbe with several access points to cater to diverse user needs.

### Python usage

It enables full control over the functionalities as well as process customization.

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
- `--hf-token`: To download the models, a [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) must be generated. Check [Hugging Face](https://huggingface.co/docs/hub/models) for further information on how to access the models.
- `--server-name`: Name of the Web Server. If empty 127.0.0.1 or 0.0.0.0 will be used.
-  `--port`: To run the Gradio app. The default is 7860.

- `--whisper-model-name`: Name of the [whisper](https://github.com/openai/whisper) model to be used. Default is `medium`.


Run the following to view all available options:
		
	autotranscript -h

## Documentation 

For further insights check the [documentation page](https://cristinaortizcruz.github.io/Test/).

## Contributions

We are happy for any interest in contributing: In order to do that, fork the repo and use merge requests to incorporate your contribution.

## Roadmap

The following milestones are planned for further releases of ScrAIbe:

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

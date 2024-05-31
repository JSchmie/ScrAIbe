# `ScrAIbe: Streamlined Conversation Recording with Automated Intelligence Based Environment` 🎙️🧠

Welcome to `ScrAIbe`, a state-of-the-art, [PyTorch](https://pytorch.org/) based multilingual speech-to-text framework designed to generate fully automated transcriptions. 

Beyond transcription, ScrAIbe supports advanced functions such as speaker diarization and speaker recognition. 🚀

Designed as a comprehensive AI toolkit, it uses multiple powerful AI models:

- **[Whisper](https://github.com/openai/whisper)**: A general-purpose speech recognition model.
- **[WhisperX](https://github.com/m-bain/whisperX)**: A faster, quantized version of Whisper for enhanced performance on CPU. ⚡
- **[Pyannote-Audio](https://github.com/pyannote/pyannote-audio)**: An open-source toolkit for speaker diarization. 🗣️

The framework utilizes a PyanNet-inspired pipeline, with the `Pyannote` library for speaker diarization and `VoxCeleb` for speaker embedding.

During post-diarization, each audio segment is processed by the OpenAI `Whisper` model in a transformer encoder-decoder structure. Initially, a CNN mitigates noise and enhances speech. Before transcription, `VoxLingua` identifies the language segment, facilitating Whisper's role in both transcription and text translation. 🌍✨

The following graphic illustrates the whole pipeline:

<div style="text-align:center;">
  <img src="./Pictures/pipeline.png#gh-dark-mode-only" style="width: 60%;" />
  <img src="./Pictures/pipeline_light.png#gh-light-mode-only" style="width: 60%;" />
</div>

## Getting Started 🚀

### Prerequisites

Before installing ScrAIbe, ensure you have the following prerequisites:

- **Python**: Version 3.9 or later.
- **PyTorch**: Version 2.0 or later.
- **CUDA**: A compatible version with your PyTorch Version if you want to use GPU acceleration.

**Note:** PyTorch should be automatically installed with the pip installer. However, if you encounter any issues, you should consider installing it manually by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

### Install ScrAIbe 

Install ScrAIbe on your local machine with ease using PyPI.

```bash
pip install scraibe
```

If you want to install the development version, you can do so by installing it from GitHub:

```bash
pip install git+https://github.com/JSchmie/ScrAIbe.git@develop
```

or from PyPI using our latest pre-release:

```bash
pip install --pre scraibe
```

Get started with ScrAIbe today and experience seamless, automated transcription and diarization.

## Usage

We've developed ScrAIbe with several access points to cater to diverse user needs.

### Python Usage

Gain full control over the functionalities as well as process customization.

```python
from scraibe import Scraibe

model = Scraibe()

text = model.autotranscribe("audio.wav")

print(f"Transcription: \n{text}")
```

The `Scraibe` class ensures the models are properly loaded. You can customize the models with various keywords:

- **Whisper Models**: Use the `whisper_model` keyword to specify models like `tiny`, `base`, `small`, `medium`, or `large` (`large-v2`, `large-v3`) depending on your accuracy and speed needs.
- **Pyannote Diarization Model**: Use the `dia_model` keyword to change the diarization model.
- **WhisperX**: Set the `whisper_type` to `"whisperX"` for enhanced performance on CPU and use their enhanced models. (Model names are the same)
- **Keyword Arguments**: A variety of different `kwargs` are available:
  - `use_auth_token`: Pass a Hugging Face token to the Pyannote backend if you want to use one of the models hosted on their Hugging Face.
  - `verbose`: Enable this to add an additional level of verbosity.
  
  In general, you should be able to input any `kwargs` that you can input in the original Whisper (WhisperX) and Pyannote Python APIs.

As input, `autotranscribe` accepts every format compatible with [FFmpeg](https://ffmpeg.org/ffmpeg-formats.html). Examples include `.mp4`, `.mp3`, `.wav`, `.ogg`, `.flac`, and many more.

To further control the pipeline of `ScrAIbe`, you can pass almost any keyword argument that is accepted by `Whisper` or `Pyannote`. For more options, refer to the documentation of these frameworks, as their keywords are likely to work here as well. 

Here are some examples regarding `diarization` (which relies on the `pyannote` pipeline):

- `num_speakers`: Number of speakers in the audio file
- `min_speakers`: Minimum number of speakers in the audio file
- `max_speakers`: Maximum number of speakers in the audio file

Then there are arguments for the transcription process, which uses the "Whisper" model:

- `language`: Specify the language ([list of supported languages](https://github.com/openai/whisper/blob/main/language-breakdown.svg))
- `task`: Can be either `transcribe` or `translate`. If `translate` is selected, the transcribed audio will be translated to English.

For example:

```python
text = model.autotranscribe("audio.wav", language="german", num_speakers = 2)
```

`Scraibe` also contains the option to just do a transcription:

```python
transcription = model.transcribe("audio.wav")
``` 

or just do a diarization:

```python
diarization = model.diarization("audio.wav")
```

Start exploring the powerful features of ScrAIbe and customize it to fit your specific transcription and diarization needs!

### Command-line usage

Next to the Pyhton interface, you can also run ScrAIbe using the command-line interface:

```bash
scraibe -f "audio.wav" --language "german" --num_speakers 2
```

For the full list of options, run:

```bash
scraibe -h
```

This will display a comprehensive list of all command-line options, allowing you to tailor ScrAIbe’s functionality to your specific needs.

## Gradio App 🌐

The Gradio App is now part of ScrAIbe-WebUI! This user-friendly interface enables you to run the model without any coding knowledge. You can easily run the app in your browser and upload your audio files, or make the framework available on your network and run it on your local machine. 🚀

All functionalities previously available in the Gradio App are now part of the ScrAIbe-WebUI. For more information and detailed instructions, visit the [ScrAIbe-WebUI GitHub repository](https://github.com/JSchmie/ScrAIbe-WebUI).

## Docker Container 🐳

ScrAIbe's Docker containers have also moved to ScrAIbe-WebUI! This option is especially useful if you want to run the model on a server or if you would like to use the GPU without dealing with CUDA.

All Docker container functionalities are now part of ScrAIbe-WebUI. For more information and detailed instructions on how to use the Docker containers, please visit the [ScrAIbe-WebUI GitHub repository](https://github.com/JSchmie/ScrAIbe-WebUI).

---

With these changes, ScrAIbe focuses on its core functionalities while the enhanced Gradio App and related Docker containers are now part of ScrAIbe-WebUI. Enjoy a more streamlined and powerful transcription experience! 🎉

## Documentation 📚

For comprehensive guides, detailed instructions, and advanced usage tips, visit our [documentation page](https://jschmie.github.io/ScrAIbe/). Here, you will find everything you need to make the most out of ScrAIbe.

### Contributions 🤝

We warmly welcome contributions from the community! Whether you’re fixing bugs, adding new features, or improving documentation, your help is invaluable. Please see our [Contributing Guidelines](./CONTRIBUTING.md) for more information on how to get involved and make your mark on ScrAIbe-WebUI.


### License 📜

ScrAIbe-WebUI is proudly open source and licensed under the GPL-3.0 license. This promotes a collaborative and transparent development process. For more details, see the [LICENSE](./LICENSE) file in this repository.

## Acknowledgments

Special thanks go to the [KIDA](https://www.kida-bmel.de/) project and the [BMEL (Bundesministerium für Ernährung und Landwirtschaft)](https://www.bmel.de/EN/Home/home_node.html), especially to the AI Consultancy Team.

---

Join us in making ScrAIbe even better! 🚀
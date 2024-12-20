# `ScrAIbe: Streamlined Conversation Recording with Automated Intelligence Based Environment` 🎙️🧠

Welcome to `ScrAIbe`, a state-of-the-art, [PyTorch](https://pytorch.org/) based multilingual speech-to-text framework designed to generate fully automated transcriptions.

Beyond transcription, ScrAIbe supports advanced functions such as speaker diarization and speaker recognition. 🚀

Designed as a comprehensive AI toolkit, it uses multiple powerful AI models:

- **[Whisper](https://github.com/openai/whisper)**: A general-purpose speech recognition model.
- **[FasterWhisper](https://github.com/guillaumekln/faster-whisper)**: An optimized version of Whisper for enhanced performance and flexibility.
- **[Pyannote-Audio](https://github.com/pyannote/pyannote-audio)**: An open-source toolkit for speaker diarization. 🗣️

The framework utilizes a PyanNet-inspired pipeline, with the `Pyannote` library for speaker diarization and `VoxCeleb` for speaker embedding.

During post-diarization, each audio segment is processed by the OpenAI `Whisper` model or its optimized counterpart `FasterWhisper`. Initially, a CNN mitigates noise and enhances speech. Before transcription, `VoxLingua` identifies the language segment, facilitating Whisper's role in both transcription and text translation. 🌍✨

---

## Getting Started 🚀

### Prerequisites

Before installing ScrAIbe, ensure you have the following prerequisites:

- **Python**: Version 3.11 or later.
- **PyTorch**: Version 2.0 or later.
- **CUDA**: A compatible version with your PyTorch version if you want to use GPU acceleration.

**Note:** PyTorch should be automatically installed with the pip installer. However, if you encounter any issues, consider installing it manually by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

---

### Install ScrAIbe

Install ScrAIbe on your local machine with ease using PyPI.

```bash
pip install scraibe
```

For the development version, install directly from GitHub:

```bash
pip install git+https://github.com/JSchmie/ScrAIbe.git@develop
```

Alternatively, use the latest pre-release:

```bash
pip install --pre scraibe
```

---

### Overview of the Pipeline

Below are images illustrating the ScrAIbe pipeline. Each step, from speaker diarization to transcription, is meticulously handled by our AI models.

<div style="text-align:center;">
  <img src="./Pictures/pipeline.png#gh-dark-mode-only" style="width: 40%;" alt="Pipeline Illustration - Dark Mode" />
  <img src="./Pictures/pipeline_light.png#gh-light-mode-only" style="width: 40%;" alt="Pipeline Illustration - Light Mode" />
</div>

---

## Usage

ScrAIbe provides flexible access points to cater to various user needs.

### Python Usage

Customize and control functionalities programmatically.

```python
from scraibe import Scraibe

model = Scraibe()

text = model.autotranscribe("audio.wav")

print(f"Transcription: \n{text}")
```

#### Customization Options

- **Whisper Models**: Use the `whisper_model` keyword for models like `tiny`, `base`, `small`, `medium`, or `large` (`large-v2`, `large-v3`, `large-v3-turbo`).
- **Pyannote Diarization Model**: Use the `dia_model` keyword for alternative diarization models.
- **FasterWhisper**: Set `whisper_type` to `"faster-whisper"` for optimized performance.

#### Keyword Arguments

- `use_auth_token`: Pass a Hugging Face token to access specific models.
- `verbose`: Enable detailed logs.

Compatible formats include `.mp4`, `.mp3`, `.wav`, `.ogg`, `.flac`, and more.

```python
text = model.autotranscribe("audio.wav", language="german", num_speakers=2)
```

---

### Command-line Usage

Run ScrAIbe via CLI:

```bash
scraibe -f "audio.wav" --language "german" --num_speakers 2
```

Get the full list of options:

```bash
scraibe -h
```

---

## Gradio App 🌐

The Gradio App has transitioned to ScrAIbe-WebUI! Run the model via a user-friendly interface in your browser. Learn more at the [ScrAIbe-WebUI GitHub repository](https://github.com/JSchmie/ScrAIbe-WebUI).

---

## Docker Container 🐳

For CLI-based operations, use the `hadr0n/scraibe` Docker container. Pull the image:

```bash
docker pull hadr0n/scraibe
```

Run CLI commands inside the container:

```bash
docker exec -it <container_id_or_name> scraibe -f /data/audio.wav --language german --num_speakers 2
```

Mount your local directory:

```bash
docker run -d --name scraibe-container -v $(pwd):/data hadr0n/scraibe
```

List all CLI options:

```bash
docker exec -it scraibe-container scraibe -h
```

---

## Documentation 📚

Explore detailed guides and advanced tips on our [documentation page](https://jschmie.github.io/ScrAIbe/).

---

## Releases

For the latest updates, including bug fixes, dependency changes, and new features, refer to the [ScrAIbe Release Notes](https://github.com/JSchmie/ScrAIbe/releases). Stay informed about improvements like the transition to `FasterWhisper`, enhanced Torch configuration, and new CLI features.

---

### Contributions 🤝

We warmly welcome contributions from the community! Whether you’re fixing bugs, adding new features, or improving documentation, your help is invaluable. Please see our [Contributing Guidelines](./CONTRIBUTING.md) for more information on how to get involved and make your mark on ScrAIbe-WebUI.

---

### License 📜

ScrAIbe-WebUI is proudly open source and licensed under the GPL-3.0 license. This promotes a collaborative and transparent development process. For more details, see the [LICENSE](./LICENSE) file in this repository.

---

## Acknowledgments

Special thanks go to the [KIDA](https://www.kida-bmel.de/) project and the [BMEL (Bundesministerium für Ernährung und Landwirtschaft)](https://www.bmel.de/EN/Home/home_node.html), especially to the AI Consultancy Team.

---

Join us in making ScrAIbe even better! 🚀
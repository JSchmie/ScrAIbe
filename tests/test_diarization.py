from os import environ

environ["AUTOT_CACHE"] = "/mnt/disk1/Projekte/ScrAIbe/tests"
# environ["PYANNOTE_CACHE"] = "/mnt/disk1/Projekte/ScrAIbe/tests/pyannote"
# environ["TORCH_HOME"] = "/mnt/disk1/Projekte/ScrAIbe/tests/torch"

from scraibe import Scraibe

scraibe = Scraibe(whisper_type = "faster-whisper", whisper_model = "tiny")
print(scraibe.autotranscribe('/mnt/disk1/Projekte/ScrAIbe/test/audio_test_1.mp4'))
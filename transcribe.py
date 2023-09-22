
from scraibe import Scraibe
model = Scraibe()

text = model.autotranscocribe('kida.mp4', num_speakers=2)

print("Transcription:\n")
print(text)

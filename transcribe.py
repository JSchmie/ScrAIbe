from autotranscript.autotranscript import AutoTranscribe

model = AutoTranscribe()

text = model.transcribe("tests/test.wav")

print(text)

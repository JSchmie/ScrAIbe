# import os
# import sys
# import traceback

# class TracePrints(object):
#   def __init__(self):    
#     self.stdout = sys.stdout
#   def write(self, s):
#     self.stdout.write("Writing %r\n" % s)
#     traceback.print_stack(file=self.stdout)

# sys.stdout = TracePrints()

# os.environ["PYANNOTE_CACHE"] = os.path.expanduser("~/PycharmProjects/autotranscript/autotranscript/models/pyannote")
# import os
 
# os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser("~/PycharmProjects/autotranscript/autotranscript/models")
# os.environ['HF_HOME'] = os.path.expanduser("~/PycharmProjects/autotranscript/autotranscript/models")


from autotranscript import AutoTranscribe

model = AutoTranscribe()

text = model.transcribe("test.mp4")

print("Transcription:\n")
print(text)


# from autotranscript.misc import *
# import os

# print(os.path.exists(CACHE_DIR))
# print(os.path.exists(WHISPER_DEFAULT_PATH))
# print(os.path.exists(PYANNOTE_DEFAULT_PATH))

# print(os.path.exists(PYANNOTE_DEFAULT_CONFIG))

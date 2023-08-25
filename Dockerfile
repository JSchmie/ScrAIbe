FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Install dependencies
WORKDIR /app

ENV TRANSFORMERS_CACHE /app/models
ENV HF_HOME /app/models
ENV AUTOT_CACHE /app/models
ENV PYANNOTE_CACHE /app/models/pyannote

COPY requirements.txt /tmp/requirements.txt
COPY autotranscript /app/autotranscript
COPY models /app/models

#RUN conda env update --name base --file /tmp/env.yml
RUN pip install -r /tmp/requirements.txt
RUN conda install -y ffmpeg 
RUN conda install -c conda-forge libsndfile
RUN pip install torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install autotranscript
RUN python -c "from autotranscript import AutoTranscribe; AutoTranscribe()"

# test:
COPY dockerpytest.py /app/dockerpytest.py
COPY test.mp4 /app/test.mp4
RUN python dockerpytest.py

RUN rm -rf /app/autotranscript
RUN rm -rf /app/test.mp4
RUN rm -rf /app/dockerpytest.py

# Expose port
EXPOSE 7860

# Run the application
CMD ["autotranscript"]
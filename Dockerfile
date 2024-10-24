#pytorch Image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Labels

LABEL maintainer="Jacob Schmieder"
LABEL email="Jacob.Schmieder@dbfz.de"
LABEL version="0.1.1.dev"
LABEL description="Scraibe is a tool for automatic speech recognition and speaker diarization. \
                    It is based on the Hugging Face Transformers library and the Pyannote library. \
                    It is designed to be used with the Whisper model, a lightweight model for automatic \
                    speech recognition and speaker diarization."
LABEL url="https://github.com/JSchmie/ScrAIbe"

# Install dependencies
WORKDIR /app
#Enviorment dependencies
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV AUTOT_CACHE=/app/models
ENV PYANNOTE_CACHE=/app/models/pyannote
#Copy all necessary files 
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md
COPY scraibe /app/scraibe

#Installing all necessary dependencies and running the application with a personalised Hugging-Face-Token
RUN apt update -y && apt upgrade -y && \
    apt install -y libsm6 libxrender1 libfontconfig1 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN conda update --all && \
    # conda install -y pip ffmpeg && \
    conda install -c conda-forge libsndfile && \
    conda clean --all -y
# RUN pip install torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860
# Run the application

ENTRYPOINT ["python3", "-m",  "scraibe.cli"]  
#pytorch Image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

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
ARG model_name=medium
#Enviorment Dependncies
ENV TRANSFORMERS_CACHE /app/models
ENV HF_HOME /app/models
ENV AUTOT_CACHE /app/models
ENV PYANNOTE_CACHE /app/models/pyannote
#Copy all necessary files 
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md
COPY models /app/models
COPY scraibe /app/scraibe
COPY setup.py /app/setup.py

#Installing all necessary Dependencies and Running the Application with a personalised Hugging-Face-Token
RUN apt update && apt-get install -y libsm6 libxrender1 libfontconfig1
RUN conda update --all

RUN conda install pip
RUN conda install -y ffmpeg 
RUN conda install -c conda-forge libsndfile
RUN pip install torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN pip install markupsafe==2.0.1 --force-reinstall

RUN python3 -m 'scraibe.cli' --whisper-model-name $model_name
# Expose port
EXPOSE 7860
# Run the application

ENTRYPOINT ["python3", "-m",  "scraibe.cli" ,"--whisper-model-name", "$model_name"]  
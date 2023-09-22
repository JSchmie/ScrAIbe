#pytorch Image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# Install dependencies
WORKDIR /app
ARG hf_token
#Enviorment Dependncies
ENV TRANSFORMERS_CACHE /app/models
ENV HF_HOME /app/models
ENV AUTOT_CACHE /app/models
ENV PYANNOTE_CACHE /app/models/pyannote
#Copy all necessary files 
COPY requirements.txt /app/requirements.txt
COPY scraibe /app/Scraibe
COPY setup.py /app/setup.py
#Installing all necessary Dependencies and Running the Application with a personalised Hugging-Face-Token
RUN conda install pip
RUN conda install -y ffmpeg 
RUN conda install -c conda-forge libsndfile
RUN pip install torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install /app/ 
RUN pip install markupsafe==2.0.1 --force-reinstall
RUN Scraibe  --hf_token $hf_token
# Expose port
EXPOSE 7860
# Run the application
ENTRYPOINT ["scraibe"]  

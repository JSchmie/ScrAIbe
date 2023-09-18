from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

import base64
from scraibe.app.qtfaststart import process
from scraibe import AutoTranscribe
import io
import subprocess as sp
import numpy as np
from scraibe.audio import SAMPLE_RATE

# Setup auto-transcript
autot = AutoTranscribe() # whisper_model="tiny", whisper_kwargs={"local" : False}

# Setup FFmpeg
PROBLEMATIC_FILE_TYPES : tuple = "mov","mp4","m4a","3gp","3g2","mj2"


# Setup Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    file = io.BytesIO(decoded).read()
    
    if filename.endswith(PROBLEMATIC_FILE_TYPES):
        # mp4 and other files need to be processed with qtfaststart
        # since theire metadata is at the end of the file
        # and we need it at the beginning
        file = process(file) 

    cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i",'pipe:',
            "-f", "s16le",
            '-hide_banner',
            '-loglevel', 'error',
            "-c", "copy",
            "-vn",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-"
        ]
    
    proc = sp.Popen(cmd, stdout=sp.PIPE, stdin=sp.PIPE)
    
    out = proc.communicate(input=file)[0]
    out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    out = np.array([out, SAMPLE_RATE])
    
    transcript = str(autot.transcribe(out))
    
    return html.Div([
        html.H5(f"File Name: {filename} \n" \
                "Transcript: \n"
                ),
        html.P(transcript)
    ])

@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server()

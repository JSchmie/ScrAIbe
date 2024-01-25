"""
This script is used to start the Gradio interface for audio transcription.
A configuration file can be passed to the script to configure the interface.
If no configuration file is passed, the default configuration is used.
The main Reason for this script is to allow the use of multiprocessing in the app.
"""

import multiprocessing
from scraibe.misc import ParseKwargs
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--server-config", type=str, default= None,
                        help="Path to the configy.yml file.")
    
parser.add_argument('--server-kwargs', nargs='*', action=ParseKwargs, default={},
                    help='Keyword arguments for the Gradio app.')

args = parser.parse_args()

if __name__ == '__main__':
    
    multiprocessing.set_start_method('spawn')

    from scraibe.app.app import app
    
    app(config = args.server_config, **args.server_kwargs)
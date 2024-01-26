"""
This script is used to start the Gradio interface for audio transcription.
A configuration file can be passed to the script to configure the interface.
If no configuration file is passed, the default configuration is used.
The main Reason for this script is to allow the use of multiprocessing in the app.
"""

import multiprocessing
from argparse import ArgumentParser, Action

class ParseKwargs(Action):
    """
    Custom argparse action to parse keyword arguments. has to bne redifined here because of multiprocessing.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            try:
                value = eval(value)
            except:
                pass
            getattr(namespace, self.dest)[key] = value

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
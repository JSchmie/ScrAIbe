"""Starts the Gradio interface for audio transcription with optional configuration.

This script, app_starter.py, initializes and runs a Gradio interface for audio 
transcription tasks. It allows users to provide a configuration file for custom 
settings. If no configuration file is specified, default settings are applied. 
The script is designed to support multiprocessing for improved performance.

Attributes:
    args (argparse.Namespace): Parsed command line arguments.

Example:
    To run the script with custom server configuration and keyword arguments:
    $ python app_starter.py --server-config path/to/config.yml --server-kwargs key1=val1 key2=val2
"""

import multiprocessing
from argparse import ArgumentParser, Action

class ParseKwargs(Action):
    """Custom action for argparse to parse keyword arguments for Gradio app configuration.

    This action parses a series of keyword arguments and converts them into a 
    dictionary, which is then used to configure the Gradio application. It 
    supports dynamic types by attempting to evaluate the argument values.

    Attributes:
        dest (str): The name of the attribute to be added to the object returned by parse_args().
    """
    def __call__(self, parser, namespace, values, option_string=None):
        """Parses keyword arguments and updates the namespace with these arguments as a dictionary.

        For each value provided, this method splits the string on the '=' character 
        to separate keys and values, attempting to evaluate the values for Python 
        literals. If evaluation fails, the raw string is used as the value.

        Args:
            parser (ArgumentParser): The ArgumentParser object that called this method.
            namespace (Namespace): An argparse.Namespace object that will be returned by parse_args().
            values (list of str): List of strings, each representing a key-value pair in 'key=value' format.
            option_string (Optional[str]): The option string that was used to invoke this action.

        Raises:
            ValueError: If any string in values does not contain the '=' character, indicating an invalid format.
        """
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
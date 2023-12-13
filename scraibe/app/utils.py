import os
import warnings
import yaml

import scraibe.app.global_var as gv


class ConfigLoader:
    def __init__(self, config):
        
        self.config = config
    
    def restore_defaults_for_keys(self, *args):
        """
        Restores specified keys to their default values, including nested keys.

        Args:
            keys (list): A list of keys or paths to keys (for nested dictionaries) to restore to default values.
                         Each key or path should be a list of keys leading to the desired key.
        """
        default_config = self.get_default_config()
        
        for key in args:
            self.apply_overrides(self.config, default_config, key)
            
            
        
    @classmethod
    def load_config(cls, yaml_path = None, **kwargs):
        """
        Load the configuration file and apply overrides.

        Args:
            yaml_path (str): Path to the YAML file containing overrides.
            **kwargs: Additional overrides as keyword arguments.

        Returns:
            Config: A Config object with the loaded configuration.
        """
        
        # Load the original configuration    
        config = cls.get_default_config()
    
        # Override with another YAML file if provided
        if yaml_path:
            with open(yaml_path, 'r') as file:
                override_config = yaml.safe_load(file)
                cls.apply_overrides(config, override_config)

        # Apply overrides from kwargs
        cls.apply_overrides(config, kwargs)
        return cls(config)
    
    @staticmethod
    def apply_overrides(orig_dict, override_dict, specific=None):
        """ Recursively apply overrides to the configuration, only for specific keys. """
        for key, value in override_dict.items():
            
            if isinstance(value, dict):
                # If the value is a dict, apply recursively
                sub_dict = orig_dict.get(key, {})
                ConfigLoader.apply_overrides(sub_dict, value, specific)
                orig_dict[key] = sub_dict
            else:
                # Apply override for this key
                if specific is None:
                    # If no specific keys are provided, update the key  
                    # If the value is not a dict, search for the key and update
                    if ConfigLoader.update_nested_key(orig_dict, key, value):
                        continue  # Key was found and updated
                    orig_dict[key] = value  # Key not found, update at this level
                
                elif key in specific:
                    # If specific keys are provided, only update if the key is in the list
                    if ConfigLoader.update_nested_key(orig_dict, specific, value):
                        continue  # Key was found and updated
                    orig_dict[specific] = value

    @staticmethod
    def update_nested_key(d, key, value):
        """ Recursively search and update the key in nested dictionary. """
        
        if key in d:
            d[key] = value
            return True
        for k, v in d.items():
            if isinstance(v, dict) and ConfigLoader.update_nested_key(v, key, value):
                return True
        return False
    
    @staticmethod
    def get_default_config():
        """ Return the default configuration. """
        with open(gv.DEFAULT_APP_CONIFG_PATH , 'r') as file:
            config = yaml.safe_load(file)
        return config
        

class AppConfig(ConfigLoader):
    
    def __init__(self, config):
        
        self.config = config
        
        self.set_global_vars_from_config()
        self.set_launch_options()
        self.set_layout_options()
        
        self.lauch = self.config.get("launch")
        self.model = self.config.get("model")
        self.advanced = self.config.get("advanced")
        self.queue = self.config.get("queue")
        self.layout = self.config.get("layout")
    
    def set_global_vars_from_config(self):
        """
        Sets the global variables from a configuration dictionary.
        
        Args:
            config (dict): A dictionary containing the parameters for the model. Modify the default parameters in the config.yml file.
        
        Returns:
            None
        
        """
    
        gv.MODEL_PARAMS = self.config.get('model')
        gv.TIMEOUT = self.config.get("advanced").get('timeout')
    
    def set_launch_options(self):
        
        launch_options = self.config.get("launch")
        
        if launch_options.get('auth').pop('auth_enabled'):
            self.config['launch']['auth'] = (launch_options.get('auth').pop('auth_username'),
                                             launch_options.get('auth').pop('password'))
        else:
            self.config['launch']['auth'] = None
    
    def set_layout_options(self):
        self.config['layout']['header'] = self.check_and_set_path(self.config['layout'], 'header')
        self.config['layout']['footer'] = self.check_and_set_path(self.config['layout'], 'footer')
        self.config['layout']['logo'] = self.check_and_set_path(self.config['layout'], 'logo')

    
    @staticmethod
    def check_and_set_path(config_item, key):
        """
        Check if the file exists at the given path. If not, try with CURRENT_PATH.
        Raise FileNotFoundError if the file still doesn't exist.
        """
        _current_path = os.path.dirname(os.path.realpath(__file__))  # Define your CURRENT_PATH

        file_path = config_item.get(key)
        if file_path is None:
            return None
        if not os.path.exists(file_path):
            new_path = os.path.join(_current_path, file_path)
            if not os.path.exists(new_path):
                warnings.warn(f"{key.capitalize()} file not found: {config_item[key]} \n" \
                              "fall back to default.")
            else:
                config_item[key] = new_path
            
        return config_item[key]
    
    
    
    
    
    
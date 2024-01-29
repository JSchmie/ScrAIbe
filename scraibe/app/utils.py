"""
utils.py

This module contains two classes, ConfigLoader and AppConfig, which are used to manage application-specific configuration settings.

The ConfigLoader class provides methods for loading a configuration file, applying overrides, and restoring default values for specified keys. It also includes methods for recursively updating nested keys and getting the default configuration.

The AppConfig class extends ConfigLoader and provides additional methods for setting global variables, launch options, and layout options from the configuration. It also includes methods for checking and setting file paths, and getting layout options.

Classes:
    ConfigLoader: Manages application-specific configuration settings.
    AppConfig: Extends ConfigLoader to provide additional methods for managing application-specific configuration settings.
"""
import os
import warnings
import yaml
from typing import Any, Dict, Optional

import scraibe.app.global_var as gv

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

class ConfigLoader:
    """A class that extends ConfigLoader to manage application-specific configuration settings.

    This class provides methods for setting global variables, launch options, and layout options from the configuration.

    Attributes:
        config (Dict[str, Any]): The current configuration settings.
        launch (Dict[str, Any]): The launch configuration settings.
        model (Dict[str, Any]): The model configuration settings.
        advanced (Dict[str, Any]): The advanced configuration settings.
        queue (Dict[str, Any]): The queue configuration settings.
        layout (Dict[str, Any]): The layout configuration settings.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initializes a new instance of the ConfigLoader class.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        
    def restore_defaults_for_keys(self, *args: str):
        """Restores specified keys to their default values, including nested keys.

        Args:
            *args (str): A list of keys or paths to keys (for nested dictionaries) to restore to default values.
                         Each key or path should be a list of keys leading to the desired key.
        """
        default_config = self.get_default_config()
        
        for key in args:
            self.apply_overrides(self.config, default_config, key)
            
            
        
    @classmethod
    def load_config(cls, yaml_path: Optional[str] = None, **kwargs: Any) -> 'ConfigLoader':
        """Load the configuration file and apply overrides.

        Args:
            yaml_path (str, optional): Path to the YAML file containing overrides.
            **kwargs: Additional overrides as keyword arguments.

        Returns:
            ConfigLoader: A ConfigLoader object with the loaded configuration.
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
    def apply_overrides(orig_dict: Dict[str, Any], override_dict: Dict[str, Any], specific: Optional[str] = None):
        """Recursively apply overrides to the configuration, only for specific keys.

        Args:
            orig_dict (Dict[str, Any]): The original dictionary.
            override_dict (Dict[str, Any]): The override dictionary.
            specific (str, optional): The specific key to override.
        """
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
        """Recursively search and update the key in nested dictionary.

        Args:
            d (Dict[str, Any]): The dictionary.
            key (str): The key to update.
            value (Any): The new value.

        Returns:
            bool: True if the key was found and updated, False otherwise.
        """
        
        if key in d:
            d[key] = value
            return True
        for k, v in d.items():
            if isinstance(v, dict) and ConfigLoader.update_nested_key(v, key, value):
                return True
        return False
    
    @staticmethod
    def get_default_config():
        """Return the default configuration.

        Returns:
            Dict[str, Any]: The default configuration.
        """
        with open(gv.DEFAULT_APP_CONIFG_PATH , 'r') as file:
            config = yaml.safe_load(file)
        return config
        

class AppConfig(ConfigLoader):
    """A class that extends ConfigLoader to manage application-specific configuration settings.

    This class provides methods for setting global variables, launch options, and layout options from the configuration.

    Attributes:
        config (dict): The current configuration settings.
        launch (dict): The launch configuration settings.
        model (dict): The model configuration settings.
        advanced (dict): The advanced configuration settings.
        queue (dict): The queue configuration settings.
        layout (dict): The layout configuration settings.
    """
    def __init__(self, config : Dict[str, Any]):
        """Initializes a new instance of the AppConfig class.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        
        self.set_global_vars_from_config()
        self.set_launch_options()
        self.set_layout_options()
        
        self.launch = self.config.get("launch")
        self.model = self.config.get("model")
        self.advanced = self.config.get("advanced")
        self.queue = self.config.get("queue")
        self.layout = self.config.get("layout")
    
    def set_global_vars_from_config(self) -> None:
        """Sets the global variables from a configuration dictionary.

        Args:
            config (dict): A dictionary containing the parameters for the model. Modify the default parameters in the config.yml file.

        Returns:
            None
        """
    
        gv.MODEL_PARAMS = self.config.get('model')
        gv.TIMEOUT = self.config.get("advanced").get('timeout')
    
    def set_launch_options(self) -> None:
        """Sets the launch options from a configuration dictionary.

        Args:
            None

        Returns:
            None
        """
        launch_options = self.config.get("launch")
        
        if launch_options.get('auth').pop('auth_enabled'):
            self.config['launch']['auth'] = (launch_options.get('auth').pop('auth_username'),
                                             launch_options.get('auth').pop('auth_password'))
        else:
            self.config['launch']['auth'] = None
    
    def set_layout_options(self) -> None:
        """Sets the layout options from a configuration dictionary.

        Args:
            None

        Returns:
            None
        """
        self.config['layout']['header'] = self.check_and_set_path(self.config['layout'], 'header')
        self.config['layout']['footer'] = self.check_and_set_path(self.config['layout'], 'footer')
        self.config['layout']['logo'] = self.check_and_set_path(self.config['layout'], 'logo')
    
    def get_layout(self) -> Dict[str, str]:
        """Gets the layout options from a configuration dictionary.

        Args:
            None

        Returns:
            dict: A dictionary containing the header and footer layout options.
        """
        if not os.path.exists(self.config['layout']['header']) and \
            self.config['layout']['header'] == "scraibe/app/header.html": 
            
            hname = os.path.join(CURRENT_PATH, "header.html")
            
            header = open(hname).read()
            
        elif not os.path.exists(self.config['layout']['header']) and self.config['layout']['header'] != "scraibe/app/header.html":
            warnings.warn(f"Header file not found: {self.config['layout']['header']} \n" \
                              "fall back to default.")
            
            hname = os.path.join(CURRENT_PATH, "header.html")
            
            header = open(hname).read()
        elif os.path.exists(self.config['layout']['header']):
            header = open(self.config['layout']['header']).read()    
        else:
            warnings.warn(f"Header file not found: {self.config['layout']['header']}")
            header = None
            
              
        if header != None: 
            if self.config['layout']['logo'] == "scraibe/app/logo.svg":
                header = header.replace("/file=logo.svg", f"/file={os.path.join(CURRENT_PATH, 'logo.svg')}")
            elif self.config['layout']['logo'] != "scraibe/app/logo.svg":
                header = header.replace("/file=logo.svg", f"/file={self.config['layout']['logo']}")
            else:
                warnings.warn(f"Logo file not found: {self.config['layout']['logo']}")
            
        
        if self.config['layout']['footer'] != None:
            if os.path.exists(self.config['layout']['footer']):
                footer = open(self.config['layout']['footer']).read()
            elif self.config['layout']['footer'] == None:
                footer = None
            else:    
                warnings.warn(f"Footer file not found: {self.config['layout']['footer']}")
        else:
            footer = None
        return {'header' : header ,
                'footer' : footer} 
    
    @staticmethod
    def check_and_set_path(config_item: dict, key: str) -> Optional[str]:
        """Check if the file exists at the given path. If not, try with CURRENT_PATH.
        Raise FileNotFoundError if the file still doesn't exist.

        Args:
            config_item (dict): The configuration item.
            key (str): The key to check in the configuration item.

        Returns:
            str: The path to the file if it exists, None otherwise.
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
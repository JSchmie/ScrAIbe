import scraibe.app.global_var as gv
import yaml

def load_config(original_config_path = gv.DEFAULT_APP_CONIFG_PATH, override_yaml_path=None, **kwargs):
    
    
    # Load the original configuration
    with open(original_config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Override with another YAML file if provided
    if override_yaml_path:
        with open(override_yaml_path, 'r') as file:
            override_config = yaml.safe_load(file)
            apply_overrides(config, override_config)

    # Apply overrides from kwargs
    apply_overrides(config, kwargs)

    return config

def apply_overrides(orig_dict, override_dict):
    """ Recursively apply overrides to the configuration. """
    for key, value in override_dict.items():
        if isinstance(value, dict):
            # If the value is a dict, apply recursively
            apply_overrides(orig_dict.get(key, {}), value)
        else:
            # If the value is not a dict, search for the key and update
            if update_nested_key(orig_dict, key, value):
                continue  # Key was found and updated
            orig_dict[key] = value  # Key not found, update at this level

def update_nested_key(d, key, value):
    """ Recursively search and update the key in nested dictionary. """
    if key in d:
        d[key] = value
        return True
    for k, v in d.items():
        if isinstance(v, dict) and update_nested_key(v, key, value):
            return True
    return False
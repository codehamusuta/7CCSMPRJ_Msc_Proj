import yaml

def load_config(config_fp = './config.yaml'):
    with open(config_fp) as f:
        config = yaml.safe_load(f)
    return config
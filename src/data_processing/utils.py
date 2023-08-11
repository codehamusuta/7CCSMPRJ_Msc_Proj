import os
import yaml
import random
import numpy as np

def load_config(config_fp = 'config.yaml'):
    with open(config_fp) as f:
        config = yaml.safe_load(f)
    return config

class SimpleRandom():
    """Generate random number"""
    instance = None
    def __init__(self,seed):
        self.seed = seed
        self.random = random.Random(seed)

    def next_rand(self,a,b):
        return self.random.randint(a,b)

    @staticmethod
    def get_instance():
        """Implements singleton pattern"""
        if SimpleRandom.instance is None:
            SimpleRandom.instance = SimpleRandom(SimpleRandom.get_seed())
        return SimpleRandom.instance

    @staticmethod
    def get_seed():
        return int(os.getenv("RANDOM_SEED", 12459))

    @staticmethod
    def set_seeds():
        np.random.seed(SimpleRandom.get_seed())
        random.seed(SimpleRandom.get_seed())
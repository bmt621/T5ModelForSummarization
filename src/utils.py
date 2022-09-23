import yaml
from yaml.loader import SafeLoader
import logging


def load_configs(filename):
    with open(filename) as f:
        data = yaml.load(f,Loader=SafeLoader)

    return data

def get_logger(logger_file):
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    stream_h = logging.StreamHandler()
    file_h = logging.FileHandler(logger_file) # logger files keeps tracks of all your logging messages

    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_h.setFormatter(format)
    file_h.setFormatter(format)

    logger.addHandler(stream_h)
    logger.addHandler(file_h)

    return logger
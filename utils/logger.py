import yaml, os
import logging
import logging.config
DEFAULT_LEVEL = logging.INFO
def log_setup(log_cfg_path='configs/logging_config.yaml'):
    if os.path.exists(log_cfg_path):
        with open(log_cfg_path, 'rt') as cfg_file:
            try:
                config = yaml.safe_load(cfg_file.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print('Error with file, using Default logging')
                logging.basicConfig(level=DEFAULT_LEVEL)
    else:
        logging.basicConfig(level=DEFAULT_LEVEL)
        print('Config file not found, using Default logging')#set up logging configuration
log_setup()

def get_logger(name):
    return logging.getLogger(name)
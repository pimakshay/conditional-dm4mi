import importlib

def instantiate_from_configs(config):
    if not 'target' in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string):
    module, cls = string.rsplit(".",1)
    return getattr(importlib.import_module(module, package=None), cls)

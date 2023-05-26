import os
import yaml
import importlib


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def check_path_recursively(save_folder, config,
                           params=["model_name", "pretrained_name", "dataset", "similarity_measure", "prompt"]):
    params = [config[i] for i in params if config.get(i) is not None]
    for i in params:
        save_folder = save_folder + "/" + i
        check_path(save_folder)
    config["save_folder"] = save_folder


def yaml_writer(path, contents):
    with open(path, "w") as f:
        yaml.dump(contents, f)


def text_file_reader(path):
    with open(path, "r") as f:
        contents = list(f.readlines())
    contents = [i.split("\n")[0] for i in contents]
    return contents


def text_file_writer(path, contents):
    with open(path, "w") as f:
        for line in contents:
            if isinstance(line, list):
                f.write(" ".join(line))
                f.write("\n")
            else:
                f.write(line + "\n")


def instantiate_attribute(path):
    module_path, attribute_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attribute_name)


def instantiate_class(path, params):
    optimizer_attribute = instantiate_attribute(path)
    return optimizer_attribute(params)

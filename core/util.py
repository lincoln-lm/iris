from platformdirs import user_config_dir
import json
import os
import logging

CONFIG_FOLDER = user_config_dir(appname="iris", appauthor="lincoln-lm")


def save_config_json(data, filename="config.json"):
    if not os.path.exists(CONFIG_FOLDER):
        os.makedirs(CONFIG_FOLDER)
    logging.info(" Saving config file %s", CONFIG_FOLDER + "/" + filename)
    with open(CONFIG_FOLDER + "/" + filename, "w") as f:
        json.dump(data, f)


def load_config_json(filename="config.json"):
    if not os.path.exists(CONFIG_FOLDER):
        os.makedirs(CONFIG_FOLDER)
    logging.info(" Loading config file %s", CONFIG_FOLDER + "/" + filename)
    if not os.path.exists(CONFIG_FOLDER + "/" + filename):
        logging.warning("Config file %s does not exist", CONFIG_FOLDER + "/" + filename)
        return None
    try:
        with open(CONFIG_FOLDER + "/" + filename, "r") as f:
            return json.load(f)
    except json.decoder.JSONDecodeError:
        logging.error(
            "Failed to process config file %s", CONFIG_FOLDER + "/" + filename
        )
        return None

import json
def load_config(path="config/lora_config.json"):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)



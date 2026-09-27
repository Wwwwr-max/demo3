import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_path(value):
    if value is None:
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _resolve_model_path(value):
    if value is None:
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    # Keep Hugging Face repository ids such as Qwen/Qwen2.5-7B-Instruct.
    if (
        not value.startswith((".", "~"))
        and "/" in value
        and not (PROJECT_ROOT / path).exists()
    ):
        return value

    return str((PROJECT_ROOT / path).resolve())


def load_config(path="config/lora_config.json"):
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    config["model"]["model_name_or_path"] = _resolve_model_path(
        config["model"]["model_name_or_path"]
    )
    config["data"]["root_dir"] = _resolve_path(
        config["data"]["root_dir"]
    )
    config["training"]["output_dir"] = _resolve_path(
        config["training"]["output_dir"]
    )

    test_config = config.get("test")
    if test_config is not None:
        test_config["adapter_path"] = _resolve_path(
            test_config.get("adapter_path")
        )

    return config


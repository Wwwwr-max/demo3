import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from config import load_config
from trainer import Trainer_t


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA or QLoRA adapter."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the same config JSON used for training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    test_config = config.get("test", {})

    adapter_path = test_config.get("adapter_path")
    if adapter_path is not None and not Path(adapter_path).exists():
        raise FileNotFoundError(
            f"LoRA adapter does not exist: {adapter_path}"
        )

    batch_size = test_config.get(
        "batch_size",
        config["training"]["per_device_eval_batch_size"],
    )

    trainer = Trainer_t(config)
    model = trainer.set_model(
        resume_adapter_path=adapter_path,
    )

    test_dataset = trainer.set_data("test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=trainer.set_collate(),
        num_workers=trainer.dataloader_num_workers,
        pin_memory=trainer.pin_memory,
        persistent_workers=trainer.dataloader_num_workers > 0,
    )

    metrics = trainer.evaluate(model, test_loader)

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

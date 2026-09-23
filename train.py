import argparse

from config import load_config
from trainer import Trainer_t


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LoRA or QLoRA model."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training config JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer_t(config)
    trainer.train()


if __name__ == "__main__":
    main()

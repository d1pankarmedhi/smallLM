import argparse

import torch

from smalllm.config import Config
from smalllm.dataset.text_dataset import create_data_loaders
from smalllm.logger import get_logger
from smalllm.model import LanguageModel
from smalllm.tokenizer import Tokenizer
from smalllm.trainer import Trainer, plot_losses

logger = get_logger(__name__)
config = Config()
tokenizer = Tokenizer()


def train():
    logger.info("Starting training mode.")
    logger.info(f"Device: {config.device}")

    # Load text data
    with open(config.train_data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    logger.info(f"Loaded training data from {config.train_data_path}.")

    train_loader, val_loader = create_data_loaders(
        data=text_data, tokenizer=tokenizer, config=config, val_split=0.1
    )

    model = LanguageModel(config).to(config.device)

    trainer = Trainer(model, train_loader, val_loader, config)
    logger.info("Trainer initialized. Beginning training...")
    train_loss, val_loss = trainer.train()
    logger.info("Training complete. Plotting losses.")
    plot_losses(train_loss, val_loss, config.eval_interval)


def generate(query: str, max_new_tokens: int = 100, temperature: float = 1.5):
    logger.info("Starting text generation mode.")

    model = LanguageModel(config)
    logger.info("LanguageModel initialized.")
    model.load_state_dict(
        torch.load(config.best_model_path, map_location=config.device)
    )
    logger.info(f"Loaded model weights from {config.best_model_path}.")
    model.to(config.device)
    model.eval()

    encoded = tokenizer.encode(query)
    logger.info(f"Encoded input query: {query}")
    input_ids = torch.tensor([encoded], device=config.device)

    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    output = tokenizer.decode(generated[0].tolist())
    logger.info("Text generation complete.")
    print(f"\nGenerated Text:\n{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or generate with your Language Model."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("train", help="Train the model")
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text using the trained model"
    )
    gen_parser.add_argument(
        "--query", type=str, required=True, help="Input text prompt"
    )
    gen_parser.add_argument(
        "--max_new_tokens", type=int, default=100, help="Number of tokens to generate"
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=1.5, help="Sampling temperature"
    )

    args = parser.parse_args()

    if args.mode == "train":
        logger.info("CLI: train mode selected.")
        train()
    elif args.mode == "generate":
        logger.info("CLI: generate mode selected.")
        generate(args.query, args.max_new_tokens, args.temperature)

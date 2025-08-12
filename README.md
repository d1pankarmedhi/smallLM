<div align="center">
<h1>SmallLM</h1>
<p>A small, GPT like Language Model</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white)

</div>

This project provides a minimal, modular, and extensible framework for training and generating text with a transformer-based GPT like language model. It is small language model with close to 14M parameters.

<div align="center">
<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>BPE Tokenization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/271fe3e1-07cb-4cbd-a3ea-1f3d32d7d2d3" width="300" /></td>
      <td><img src="https://github.com/user-attachments/assets/dbcad90f-0d78-4407-a48d-4973027fb9b2" width="300" /></td>
    </tr>
  </tbody>
</table>
</div>

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/smallLM.git
   cd smallLM
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Your Dataset

- Place your training text file (e.g., `data/data.txt`) in the project directory.
- Update the `train_data_path` in `smalllm/config/config.py` if needed.

### 2. Train the Model

```sh
python main.py train
```

- Training progress and losses will be logged and plotted.
- The best model checkpoint will be saved to the path specified in your config.

### 3. Generate Text

```sh
python main.py generate --query "Once upon a time" --max_new_tokens 100 --temperature 1.2
```

- `--query`: The prompt to start generation.
- `--max_new_tokens`: Number of tokens to generate (default: 100).
- `--temperature`: Sampling temperature (default: 1.5).

---

## Project Structure

```
smallLM/
│
├── main.py                      # CLI entry point
├── README.md
├── requirements.txt
│
└── smalllm/
    ├── config/
    │   └── config.py            # Configuration class
    ├── dataset/
    │   └── text_dataset.py      # Dataset and DataLoader utilities
    ├── logger.py                # Logger utility
    ├── model/
    │   ├── __init__.py
    │   └── lm.py                # LanguageModel definition
    ├── tokenizer.py             # Tokenizer class
    └── trainer.py               # Training and plotting utilities
```

## Customization

- **Model:** Edit `smalllm/model/lm.py` or adjust hyperparameters in `smalllm/config/config.py`.
- **Tokenizer:** Swap out or extend `smalllm/tokenizer.py` for different tokenization strategies.
- **Dataset:** Use any plain text file; the loader will handle splitting and batching.


## License

MIT License

## Acknowledgements

Inspired by GPT and nanoGPT projects.

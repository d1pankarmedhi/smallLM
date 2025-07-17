import tiktoken


class Tokenizer:
    """BPE Tokenizer"""

    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")  # bpe tokenizer

    def encode(self, text):
        return self.encoder.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens):
        return self.encoder.decode(tokens)

    @property
    def vocab_size(self):
        return self.encoder.n_vocab  # 50257

    @property
    def eot_token(self):
        return self.encoder.eot_token  # 50256

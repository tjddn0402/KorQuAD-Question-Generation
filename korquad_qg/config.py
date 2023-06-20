from typing import NamedTuple


class QGConfig(NamedTuple):
    dataset: str = "/opt/ml/tests/level2_nlp_mrc-nlp-03/data/train_dataset"

    gpt_model_hub_name: str = "taeminlee/kogpt2"
    vocab_path: str = "tokenizer/vocab.json"
    tokenizer_merges_path: str = "tokenizer/merges.txt"

    max_sequence_length: int = 512

    epochs: int = 21
    lr: float = 5e-5
    train_batch_size: int = 16
    eval_batch_size: int = 16

    output_dir: str = "outputs/"

    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    train_log_interval: int = 50
    validation_interval: int = 500
    save_interval: int = 1000
    random_seed: int = 0

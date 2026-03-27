import argparse
from typing import List, Optional

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration


NEW_TOKENS: List[str] = [
    "<|attribute_begin|>"
    "<|fundus_nodr|>",
    "<|fundus_milddr|>",
    "<|fundus_moderatedr|>",
    "<|fundus_severedr|>",
    "<|fundus_pdr|>",
    "<|attribute_end|>"
]


def _load_tokens_from_file(token_file: str) -> List[str]:
    """
    Load tokens from a text file.
    - One token per line
    - Empty lines are ignored
    - Lines starting with '#' are treated as comments
    """
    tokens: List[str] = []
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            tokens.append(token)
    return tokens


def add_special_tokens(model_path: str, token_file: Optional[str] = None) -> None:
    tokens_to_add = _load_tokens_from_file(token_file) if token_file else NEW_TOKENS
    if len(tokens_to_add) == 0:
        raise ValueError("No tokens to add. Please provide non-empty NEW_TOKENS or token_file.")

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    old_vocab_size = len(tokenizer)
    old_special_tokens = list(getattr(tokenizer, "additional_special_tokens", []))

    # Add tokens to tokenizer tail only (no overwrite of existing ids).
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    new_vocab_size = len(tokenizer)
    new_special_tokens = list(getattr(tokenizer, "additional_special_tokens", []))

    debug = False
    if debug:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
        old_embed_size = model.get_input_embeddings().num_embeddings

        print(f"old_vocab_size={old_vocab_size}, old_embed_size={old_embed_size}")
        print(f"num_added={num_added}")
        print(f"new_vocab_size={new_vocab_size}")
        print(f"special_tokens_before({len(old_special_tokens)}): {old_special_tokens}")
        print(f"special_tokens_after({len(new_special_tokens)}): {new_special_tokens}")

        for token in tokens_to_add:
            tid = tokenizer.convert_tokens_to_ids(token)
            is_tail = tid >= old_vocab_size
            print(f"{token}: id={tid}, is_new_tail_id={is_tail}")
    
    return processor


if __name__ == "__main__":
    model_path = "/your_path/Qwen/Qwen2.5-VL-7B-Instruct"
    token_file = "datasets/demo/attribute_list.txt"
    new_processor = add_special_tokens(model_path, token_file=token_file)

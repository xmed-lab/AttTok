import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor


from datasets.add_attribute import add_special_tokens


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    """Return the first index of `needle` in `haystack` from `start`, else -1."""
    if len(needle) == 0:
        return -1
    end = len(haystack) - len(needle) + 1
    for i in range(start, max(start, end)):
        if haystack[i : i + len(needle)] == list(needle):
            return i
    return -1


class AttributeDataset(Dataset):
    """
    Qwen2.5-VL dataset for supervised fine-tuning.

    Label policy:
    - From assistant response start to sequence end participates in loss.
    - Prompt/system/user/template tokens are set to -100.
    - If assistant boundary is not found, this sample is fully masked.
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        processor: AutoProcessor,
        max_length: int = 1024,
    ) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        # Qwen chat template markers.
        self.assistant_start_ids = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        self.invalid_supervision_count = 0
        self.invalid_class_label_count = 0

    def __len__(self) -> int:
        return len(self.data)

    def _build_messages(self, sample: Dict[str, str]) -> List[Dict[str, object]]:
        # Keep message schema explicit to avoid processor ambiguity across versions.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["input"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["output"]}],
            },
        ]

    def _build_labels(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Build labels that supervise from assistant response start to sequence end.

        Returns labels with same length as input_ids:
        - response-start .. sequence-end tokens: token id
        - others: -100
        """
        labels = torch.full_like(input_ids, -100)
        ids = input_ids.tolist()

        # Find "<|im_start|>assistant\n"
        start_anchor = _find_subsequence(ids, self.assistant_start_ids)
        if start_anchor < 0:
            self.invalid_supervision_count += 1
            return labels

        response_start = start_anchor + len(self.assistant_start_ids)
        if response_start >= len(ids):
            self.invalid_supervision_count += 1
            return labels

        labels[response_start:] = input_ids[response_start:]
        return labels

    def _extract_class_token(self, output_text: str) -> str:
        """
        Extract class special token from output text.
        Expected format example:
        "<|attribute_begin|><|fundus_nodr|><|attribute_end|> No DR"
        """
        text = output_text.strip()
        begin_token = "<|attribute_begin|>"
        end_token = "<|attribute_end|>"

        begin_pos = text.find(begin_token)
        end_pos = text.find(end_token)

        if begin_pos < 0 or end_pos < 0 or end_pos <= begin_pos:
            raise ValueError(
                f"Output must contain '{begin_token}...{end_token}': {output_text}"
            )

        inner = text[begin_pos + len(begin_token) : end_pos].strip()
        if not inner:
            raise ValueError(f"No class token found between attribute markers: {output_text}")

        # The class token should be the first token inside marker span.
        class_token = inner.split()[0]
        if not (class_token.startswith("<|") and class_token.endswith("|>")):
            raise ValueError(
                f"Invalid class token between attribute markers: {class_token} in {output_text}"
            )
        return class_token

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item["img_path"])

        image = Image.open(img_path).convert("RGB")
        class_token = self._extract_class_token(item["output"])
        class_label = self.processor.tokenizer.convert_tokens_to_ids(class_token)
        if class_label == self.processor.tokenizer.unk_token_id:
            self.invalid_class_label_count += 1
            raise ValueError(
                f"Class token {class_token} is unknown. "
                "Please ensure add_attribute.py tokens are added to this tokenizer."
            )

        messages = self._build_messages(item)
        text_full = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Single-sample encode (no padding here; collator will pad the batch).
        inputs = self.processor(
            text=[text_full],
            images=[image],
            videos=None,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        labels = self._build_labels(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "class_label": torch.tensor(class_label, dtype=torch.long),
            # Keep sample-level leading dimension; collator concatenates on dim=0.
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }


@dataclass
class AttributeQwenDataCollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        class_labels = [f["class_label"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [f["image_grid_thw"] for f in features]

        # Use tokenizer pad token for input_ids only.
        pad_id: Optional[int] = self.processor.tokenizer.pad_token_id # <|endoftext|>
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id # <|im_end|>
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

        # Important: pad attention_mask directly; do not re-infer from input_ids.
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        class_labels_tensor = torch.stack(class_labels, dim=0)

        pixel_values_cat = torch.cat(pixel_values, dim=0)
        image_grid_thw_cat = torch.cat(image_grid_thw, dim=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "class_label": class_labels_tensor,
            "pixel_values": pixel_values_cat,
            "image_grid_thw": image_grid_thw_cat,
        }

def get_attribute_dataset(model_path, json_path, image_dir, token_file, batch_size=5):
    processor = add_special_tokens(model_path, token_file=token_file)
    dataset = AttributeDataset(
        json_path=json_path,
        image_dir=image_dir,
        processor=processor,
        max_length=1024,
    )
    collator = AttributeQwenDataCollator(processor=processor)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collator)
    return dataloader, dataset, processor


if __name__ == "__main__":
    model_path = "your_path/Qwen/Qwen2.5-VL-7B-Instruct"
    json_path = "datasets/demo/demo_attribute.json"
    image_dir = "datasets/demo"
    token_file = "datasets/demo/attribute_list.txt"

    dataloader, dataset, processor = get_attribute_dataset(model_path, json_path, image_dir, token_file, batch_size=5)

    for batch in dataloader:
        print("input_ids:", tuple(batch["input_ids"].shape))
        print("attention_mask:", tuple(batch["attention_mask"].shape))
        print("labels:", tuple(batch["labels"].shape))
        print("class_label:", batch["class_label"].tolist())
        print("pixel_values:", tuple(batch["pixel_values"].shape))
        print("image_grid_thw:", tuple(batch["image_grid_thw"].shape))

        # Quick consistency checks for debug.
        supervised_counts = (batch["labels"] != -100).sum(dim=1)
        print("supervised_tokens_per_sample:", supervised_counts.tolist())

        print("the number of padded tokens is ", (batch["input_ids"] == processor.tokenizer.pad_token_id).sum(dim=1))

        bsz = batch["input_ids"].shape[0]
        for i in range(bsz):
            response_ids = batch["input_ids"][i][batch["labels"][i] != -100]
            print(f"Sample[{i}] decoded response: {processor.decode(response_ids.tolist())}The ids are {response_ids.tolist()}")
            class_id = batch["class_label"][i].item()
            print(f"Sample[{i}] class_label_id: {class_id}, token: {processor.tokenizer.decode([class_id])}")
            print("-"*10)

        break

    print("invalid_supervision_count:", dataset.invalid_supervision_count)
    print("invalid_class_label_count:", dataset.invalid_class_label_count)

    print("--------------------------------")
    additional_special_tokens = list(getattr(processor.tokenizer, "additional_special_tokens", []))
    print("additional_special_tokens:", additional_special_tokens)
    print(f"id for additional_special_tokens: {processor.tokenizer.convert_tokens_to_ids(additional_special_tokens)}")
    
    print("--------------------------------")
    all_toks = ['<|im_end|>', '<|endoftext|>', '<|im_start|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']
    print(all_toks)
    print(f"id for all_toks: {processor.tokenizer.convert_tokens_to_ids(all_toks)}")
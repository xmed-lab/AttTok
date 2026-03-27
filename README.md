

## 📚 ATTTOK: MARRYING ATTRIBUTE TOKENS WITH GENERATIVE PRE-TRAINED VISION-LANGUAGE MODELS TOWARDS MEDICAL IMAGE UNDERSTANDING (ICLR 26)



![Pipeline Diagram](figures/pipeline.jpg)

## 🔨 Code Overview

### 📖 Datasets

- Base Dataset

    In [`src/datasets/base_dataset.py`](./codes/src/datasets/base_dataset.py), we implement a base dataset using `torch.utils.data.Dataset` for training Qwen2.5-VL.  
    Based on it, you can plug in your own data augmentation pipeline to enable online image augmentations during training.

    **Debug command**
    ```bash
    python -m datasets.base_dataset
    ```

- Attribute Dataset

    In [`src/datasets/attribute_dataset.py`](./codes/src/datasets/attribute_dataset.py), VQA samples with predefined attributes are parsed to produce the "class_label".
    Demo JSON files and attribute lists are provided in `src/datasets/demo/`.


    **Debug command**
    ```bash
    python -m datasets.attribute_dataset_dataset
    ```

**Note**: The five images in the demo folder are solely for fast debugging purposes, and their labels are randomly assigned. Please replace them with your actual training data for real experiments.


### 🔧 Training

- Training with Single GPU 

    Please use [src/train/train_att_singlegpu.py](./codes/src/train/train_att_singlegpu.py) to train models. Before training, you should prepare your dataset with predefined attributes (see provided demo in [src/datasets/demo](./codes/src/datasets/demo/)) and define model and dataset paths in [src/train/train_att_singlegpu.py](./codes/src/train/train_att_singlegpu.py).

    **Command**
    ```bash
    python -m train.train_att_singlegpu
    ```


## 📝 Citation

```bibtex
@inproceedings{
wang2026atttok,
title={AttTok: Marrying Attribute Tokens with Generative Pre-trained Vision-Language Models towards Medical Image Understanding},
author={Hualiang Wang and Xinyue Xu and Lehan Wang and Bin Pu and Xiaomeng Li},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=UjSoF5CM09}
}
```

## 🙏 Code Acknowledgments

During the development of this project, we were inspired and supported by the following outstanding open-source projects, and we would like to express our sincere gratitude to them: [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune), [transformers](https://github.com/huggingface/transformers), and [LlamaFactory](https://github.com/hiyouga/LlamaFactory)



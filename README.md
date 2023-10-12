# Dior-CVAE

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official code for the paper [Dior-CVAE: Pre-trained Language Models and Diffusion Priors for Variational Dialog Generation](https://arxiv.org/abs/2305.15025.pdf) by Tianyu Yang, Thy Thy Tran, Iryna Gurevych

### Usage

#### Prepare the dataset
##### Public Dataset
You need to download and preprocess the dataset in the following steps:

1. Download the datasets, including `DailyDialog` and `PersonaChat`

   ```shell
   bash preprocess/get_data.sh
   ```

2. Preprocess the dataset coarsely

   ```shell
   bash preprocess/process.sh
   ```

3. Preprocess the dataset into jsonl format

   ```shell
   python preprocess.py
   ```

   Note to adapt the relevant path in the code.

##### Custom Dataset
You need to  prepare the dataset in the format of jsonl. Each line is a json like:
```
{'source': The prefix of the story, 'target': The main body of the story}
```

#### Training
For unconditional generation, run the codes with:
```
python main.py --train_file [path to training set] --valid_file [path to valid set] --per_gpu_train_batch_size 16 --model_name [config info of this training] --cycle_annealing
```

For conditional generation, run the codes with:
```
python main.py --train_file [path to training set] --valid_file [path to valid set] --dataset_type wp --per_gpu_train_batch_size 16 --model_name [config info of this training] --cycle_annealing
```

#### Generation
DELLA is available for all kinds of decoding strategy. For beam search (the number of beams is default as 10), run:
```
python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --num_beams 10
```
For greedy decoding, run:
```
python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --greedy_decoding
```
For top-k, top-p sampling, run:
```
python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --top_k 50 --top_p 0.9
```



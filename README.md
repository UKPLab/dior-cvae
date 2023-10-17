# Dior-CVAE: Pre-trained Language Models and Diffusion Priors for Variational Dialog Generation

This is the official code for the paper [Dior-CVAE: Pre-trained Language Models and Diffusion Priors for Variational Dialog Generation](https://arxiv.org/abs/2305.15025.pdf) by Tianyu Yang, Thy Thy Tran, Iryna Gurevych

Please use the following citation:

```
@InProceedings{yang:2023:EMNLP,
  title     = {Dior-CVAE: Pre-trained Language Models and Diffusion Priors for Variational Dialog Generation},
  author    = {Yang, Tianyu and Tran, Thy Thy and Gurevych, Iryna},
  journal   = {arXiv preprint arXiv:2305.15025},
  year={2023}
}
```

> **Abstract:**
> Current variational dialog models have employed pre-trained language models (PLMs) to parameterize the likelihood and posterior distributions. However, the Gaussian assumption made on the prior distribution is incompatible with these distributions, thus restricting the diversity of generated responses.These models also suffer from posterior collapse, i.e., the decoder tends to ignore latent variables and directly access information captured in the encoder through the cross-attention mechanism. In this work, we propose Dior-CVAE, a hierarchical conditional variational autoencoder (CVAE) with diffusion priors to address these challenges. with an informative prior parameterized by a diffusion model.
> We employ a diffusion model to increase the complexity of the prior distribution and its compatibility to the distributions produced by a PLM. Also, we propose memory dropout to the cross-attention mechanism, which actively encourages the use of latent variables for response generation. Overall, experiments across two commonly-used open-domain dialog datasets show that our method can generate more diverse responses without large-scale dialog pre-training.


Contact person: Tianyu Yang,  yang@ukp.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Project structure

* `data/` -- this folder is used to contain all the needed data.
* `generation_output/` -- this folder is used to contain the generation results.
* `tensorboard/` -- this folder is used to contain the tensorboard record.
* `preprocess/` -- this folder contains the code for preprocessing the data.
* `improved_diffusion/` -- this folder contains the code the diffusion model.
* `diffusion/` -- this folder contains the code for the denoising network.
* `checkpoints/` -- this folder is used to save the checkpoint weights during the training process

## Requirements

* Denpendent packages information can be seen in the `requirements.txt`

* Install the environment through
  
  ```
  pip install -r requirements.txt
  ```

## Running the experiments
### Prepare the public datasets
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
### Prepare the custom Dataset
You need to  prepare the dataset in the format of jsonl. Each line is a json like:
```
{'source': The dialog context, 'target': The response}
```


### Training
- Run the codes with:

  ```shell
  python main.py --train_file [path to training set] --valid_file [path to valid set] --dataset_type wp --per_gpu_train_batch_size 16 --model_name [config info of this training] --cycle_annealing --diffusion_prior --pretrained_model facebook/bart-base --bart
  ```

### Generation
- For beam search (the number of beams is default as 10), run:

    ```shell
    python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --num_beams 10 --diffusion_prior --pretrained_model facebook/bart-base --bart
    ```
- For greedy decoding, run:

    ```shell
    python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --greedy_decoding --diffusion_prior --pretrained_model facebook/bart-base --bart
    ```
- For top-k, top-p sampling, run:

    ```shell
    python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --top_k 50 --top_p 0.9 --diffusion_prior --pretrained_model facebook/bart-base --bart
    ```
### Evaluation

- Run the evaluation code

  ```shell
  python evaluation.py
  ```

  Note to adapt the relevant path in the code.

## Reproduce

The information of the device used in the experiments is as below

```
Linux version 5.15.0-83-generic (buildd@lcy02-amd64-027) (gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0, GNU ld (GNU Binutils for Ubuntu) 2.38) #92-Ubuntu SMP Mon Aug 14 09:30:42 UTC 2023
GPU: Tesla V100-SXM3-32GB
Mem: 1510G
CPU: Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz
```



## Reference

- [DELLA](https://github.com/OpenVLG/DELLA)
- [improved diffusion](https://github.com/openai/improved-diffusion)


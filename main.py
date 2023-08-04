import argparse
import logging
import os
import json
import torch
import random
import numpy as np
import time

from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

from dataset import VAEDataset, WPDataset
from train import train, valid, generate

from model import Della
from modeling_bart import BartDella

from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='./data/yelp/yelp.train.txt', type=str,
                        help="Data path for training.")
    parser.add_argument("--valid_file", default='./data/yelp/yelp.train.txt', type=str,
                        help="Data path for valid")
    parser.add_argument("--test_file", default='./data/yelp/yelp.train.txt', type=str,
                        help="Data path for test")
    parser.add_argument("--pretrained_model", type=str, default='gpt2', 
                        help="Pretrained model to be loaded")
    parser.add_argument("--dataset_type", type=str, default='vae', choices=['vae', 'wp'], 
                        help="Dataset type")
    parser.add_argument("--output_dir", default='./checkpoints', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_name", default='della', type=str,
                        help="The model name")
    parser.add_argument("--generation_output_dir", default='./generation_output', type=str,
                        help="The output directory where the log will be written.")
    # Other parameters\
    parser.add_argument("--load_epoch", default=None, type=int, help="the epochs of trained model to load")
    parser.add_argument("--epochs", default=40, type=int, help="total epochs")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,help="Batch size per GPU for training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--kl_threshold", default=0, type=float,
                        help="The threshold of the minimum KL value, default as 0")
    parser.add_argument("--latent_size", default=32, type=int,
                        help="The dimension of latent space")
    parser.add_argument("--latent_lmf_rank", default=4, type=int,
                        help="latent size")
    parser.add_argument("--max_length", default=200, type=int,
                        help="Max length for generation")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument('--log_step', type=int, default=100,
                        help="Steps for logging")
    parser.add_argument('--num_beams', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--greedy_decoding', action='store_true',
                        help="Choose to use greedy decoding")
    parser.add_argument('--top_k', type=int, default=-1, help='Set top k')
    parser.add_argument('--top_p', type=float, default=0.9, help='Set top p')
    parser.add_argument('--repetition_penalty', type=float, default=3.1)
    parser.add_argument('--model_parallel', action='store_true', 
                        help="Choose to use model parallel, mapping the layers to different devices")
    parser.add_argument('--eval', action='store_true', help='Choose to eval the model')
    parser.add_argument('--eval_metrics', action='store_true',
                        help="Choose to eval the metrics for representation learning")
    parser.add_argument('--generation', action='store_true', help='Choose to generate')
    parser.add_argument('--use_scheduler', action='store_true',
                        help="Choose to use lr scheduler")
    parser.add_argument('--cycle_annealing', action='store_true',
                        help="Choose to use cycle annealing")
    parser.add_argument('--cycle_iters', type=int, default=2,
                        help="Set the iters for cycle annealing")
    parser.add_argument('--sample_times', type=int, default=30,
                        help="The total times of sample when computing PPL with importance weighted sampling")
    parser.add_argument('--use_bow', action='store_true',
                        help="Choose to use bow loss")
    parser.add_argument('--bow_weight',type=float, default=0.2,
                        help="Set the weight of bow loss term")
    parser.add_argument("--begin_layer", default=None, type=int,
                        help="The beginning layer to consider the latent vector, default as the first layer of model")
    parser.add_argument("--end_layer", default=None, type=int,
                        help="The end layer to consider the latent vector, default as the last layer of model")
    parser.add_argument("--bart", action='store_true', default=False)
    parser.add_argument('--diffusion_prior', '-diffusion_prior', action='store_true', default=False,
              help="Whether use the denoising diffusion model to build the prior distribution")
    parser.add_argument('--sde_type', '-sde_type', type=str, choices=['geometric_sde', 'vpsde', 'sub_vpsde', 'vesde'],
              default='vpsde')
    parser.add_argument('--beta_end', '-beta_end', type=float, default=20.0)
    parser.add_argument('--beta_start', '-beta_start', type=float, default=0.1)
    parser.add_argument('--sigma2_0', '-sigma2_0', type=float, default=0.0)
    parser.add_argument('--time_eps', '-time_eps', type=float, default=0.01)

    parser.add_argument('--diffusion_steps', '-diffusion_step', type=int, default=50)
    parser.add_argument('--learn_sigma', '-learn_sigma', action='store_true', default=True)
    parser.add_argument('--sigma_small', '-sigma_small', action='store_true', default=False)
    parser.add_argument('--noise_schedule', '-noise_schedule', choices=['cosine', 'linear'], default='linear')
    parser.add_argument('--use_kl', '-use_kl', action='store_true', default=False)
    parser.add_argument('--use_ddim', '-use_ddim', action='store_true', default=False)
    parser.add_argument('--clip_denoised', '-clip_denoised', action='store_true', default=True)
    parser.add_argument('--w', '-w', type=float, default=0.1)
    parser.add_argument('--predict_xstart', '-predict_xstart', action='store_true', default=False)
    parser.add_argument('--rescale_timesteps', '-rescale_timesteps', action='store_true', default=True)
    parser.add_argument('--rescale_learned_sigmas', '-rescale_learned_sigmas', action='store_true', default=True)
    parser.add_argument('--timestep_respacing', '-timestep_respacing', type=str, default='')
    parser.add_argument('--schedule_sampler', '-schedule_sampler', choices=['uniform', 'loss-second-moment'], default='uniform')
    args = parser.parse_args()
    return args

def prepare(args):
    torch.set_num_threads(3)

    if not args.eval and not args.generation:
        os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
        json.dump(args.__dict__, open(os.path.join(
            args.output_dir, args.model_name, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    if args.no_cuda:
        args.n_gpu = 1
    else:
        args.n_gpu = torch.cuda.device_count()
    args.batch_size = args.per_gpu_train_batch_size * args.n_gpu
    
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    if args.no_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0')


def init_para_frompretrained_bart(model, bart):
    logger.info('load bart pretrained model parameters')
    model.encoder.embed_positions.weight = bart.encoder.embed_positions.weight
    model.encoder.embed_tokens.weight = bart.encoder.embed_tokens.weight
    model.encoder.layernorm_embedding.weight = bart.encoder.layernorm_embedding.weight
    model.encoder.layernorm_embedding.bias = bart.decoder.layernorm_embedding.bias
    for i in range(len(bart.encoder.layers)):
        model.encoder.layers[
            i].fc1.weight = bart.encoder.layers[
            i].fc1.weight
        model.encoder.layers[
            i].fc1.bias = bart.encoder.layers[
            i].fc1.bias
        model.encoder.layers[
            i].fc2.weight = bart.encoder.layers[
            i].fc2.weight
        model.encoder.layers[
            i].fc2.bias = bart.encoder.layers[
            i].fc2.bias
        model.encoder.layers[
            i].final_layer_norm.weight = bart.encoder.layers[
            i].final_layer_norm.weight
        model.encoder.layers[
            i].final_layer_norm.bias = bart.encoder.layers[
            i].final_layer_norm.bias
        model.encoder.layers[
            i].self_attn.k_proj.weight = bart.encoder.layers[
            i].self_attn.k_proj.weight
        model.encoder.layers[
            i].self_attn.k_proj.bias = bart.encoder.layers[
            i].self_attn.k_proj.bias
        model.encoder.layers[
            i].self_attn.q_proj.weight = bart.encoder.layers[
            i].self_attn.q_proj.weight
        model.encoder.layers[
            i].self_attn.q_proj.bias = bart.encoder.layers[
            i].self_attn.q_proj.bias
        model.encoder.layers[
            i].self_attn.v_proj.weight = bart.encoder.layers[
            i].self_attn.v_proj.weight
        model.encoder.layers[
            i].self_attn.v_proj.bias = bart.encoder.layers[
            i].self_attn.v_proj.bias
        model.encoder.layers[
            i].self_attn.out_proj.weight = bart.encoder.layers[
            i].self_attn.out_proj.weight
        model.encoder.layers[
            i].self_attn.out_proj.bias = bart.encoder.layers[
            i].self_attn.out_proj.bias
        model.encoder.layers[
            i].self_attn_layer_norm.weight = bart.encoder.layers[
            i].self_attn_layer_norm.weight
        model.encoder.layers[
            i].self_attn_layer_norm.bias = bart.encoder.layers[
            i].self_attn_layer_norm.bias
    model.decoder.embed_positions.weight = bart.decoder.embed_positions.weight
    model.decoder.embed_tokens.weight = bart.decoder.embed_tokens.weight
    model.decoder.layernorm_embedding.weight = bart.decoder.layernorm_embedding.weight
    model.decoder.layernorm_embedding.bias = bart.decoder.layernorm_embedding.bias
    for i in range(len(bart.decoder.layers)):
        model.decoder.layers[
            i].fc1.weight = bart.decoder.layers[
            i].fc1.weight
        model.decoder.layers[
            i].fc1.bias = bart.decoder.layers[
            i].fc1.bias
        model.decoder.layers[
            i].fc2.weight = bart.decoder.layers[
            i].fc2.weight
        model.decoder.layers[
            i].fc2.bias = bart.decoder.layers[
            i].fc2.bias
        model.decoder.layers[
            i].final_layer_norm.weight = bart.decoder.layers[
            i].final_layer_norm.weight
        model.decoder.layers[
            i].final_layer_norm.bias = bart.decoder.layers[
            i].final_layer_norm.bias
        model.decoder.layers[
            i].self_attn.k_proj.weight = bart.decoder.layers[
            i].self_attn.k_proj.weight
        model.decoder.layers[
            i].self_attn.k_proj.bias = bart.decoder.layers[
            i].self_attn.k_proj.bias
        model.decoder.layers[
            i].self_attn.q_proj.weight = bart.decoder.layers[
            i].self_attn.q_proj.weight
        model.decoder.layers[
            i].self_attn.q_proj.bias = bart.decoder.layers[
            i].self_attn.q_proj.bias
        model.decoder.layers[
                i].self_attn.v_proj.weight = \
            bart.decoder.layers[
                i].self_attn.v_proj.weight
        model.decoder.layers[
            i].self_attn.v_proj.bias = bart.decoder.layers[
            i].self_attn.v_proj.bias
        model.decoder.layers[
            i].self_attn.out_proj.weight = bart.decoder.layers[
            i].self_attn.out_proj.weight
        model.decoder.layers[
            i].self_attn.out_proj.bias = bart.decoder.layers[
            i].self_attn.out_proj.bias
        model.decoder.layers[
            i].self_attn_layer_norm.weight = bart.decoder.layers[
            i].self_attn_layer_norm.weight
        model.decoder.layers[
            i].self_attn_layer_norm.bias = bart.decoder.layers[
            i].self_attn_layer_norm.bias

        model.decoder.layers[
                i].encoder_attn.k_proj.weight = \
            bart.decoder.layers[
                i].encoder_attn.k_proj.weight
        model.decoder.layers[
            i].encoder_attn.k_proj.bias = bart.decoder.layers[
            i].encoder_attn.k_proj.bias
        model.decoder.layers[
                i].encoder_attn.q_proj.weight = \
            bart.decoder.layers[
                i].encoder_attn.q_proj.weight
        model.decoder.layers[
                i].encoder_attn.q_proj.bias = \
            bart.decoder.layers[
                i].encoder_attn.q_proj.bias
        model.decoder.layers[
                i].encoder_attn.v_proj.weight = \
            bart.decoder.layers[
                i].encoder_attn.v_proj.weight
        model.decoder.layers[
                i].encoder_attn.v_proj.bias = \
            bart.decoder.layers[
                i].encoder_attn.v_proj.bias
        model.decoder.layers[
                i].encoder_attn.out_proj.weight = \
            bart.decoder.layers[
                i].encoder_attn.out_proj.weight
        model.decoder.layers[
                i].encoder_attn.out_proj.bias = \
            bart.decoder.layers[
                i].encoder_attn.out_proj.bias
        model.decoder.layers[
            i].encoder_attn_layer_norm.weight = bart.decoder.layers[
            i].encoder_attn_layer_norm.weight
        model.decoder.layers[
            i].encoder_attn_layer_norm.bias = bart.decoder.layers[
            i].encoder_attn_layer_norm.bias


def init_para_frompretrained(model, gpt2):
    logger.info('load gpt2 pretrained model parameters')
    model = model.encoder
    model.wte.weight = gpt2.wte.weight
    model.wpe.weight = gpt2.wpe.weight

    for i in range(len(gpt2.h)):
        model.h[i].ln_1.weight = gpt2.h[i].ln_1.weight
        model.h[i].ln_1.bias = gpt2.h[i].ln_1.bias
        model.h[i].attn.c_attn.weight = gpt2.h[i].attn.c_attn.weight
        model.h[i].attn.c_attn.bias = gpt2.h[i].attn.c_attn.bias
        model.h[i].attn.c_proj.weight = gpt2.h[i].attn.c_proj.weight
        model.h[i].attn.c_proj.bias = gpt2.h[i].attn.c_proj.bias
        model.h[i].ln_2.weight = gpt2.h[i].ln_2.weight
        model.h[i].ln_2.bias = gpt2.h[i].ln_2.bias
        model.h[i].mlp.c_fc.weight = gpt2.h[i].mlp.c_fc.weight
        model.h[i].mlp.c_fc.bias = gpt2.h[i].mlp.c_fc.bias
        model.h[i].mlp.c_proj.weight = gpt2.h[i].mlp.c_proj.weight
        model.h[i].mlp.c_proj.bias = gpt2.h[i].mlp.c_proj.bias

    model.ln_f.weight = gpt2.ln_f.weight
    model.ln_f.bias = gpt2.ln_f.bias

def prepare_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if '<s>' not in tokenizer.vocab:
        tokenizer._add_tokens(['<s>'])
    if '</s>' not in tokenizer.vocab:
        tokenizer._add_tokens(['</s>'])
    if not args.bart:
        tokenizer.pad_id = 50256
    tokenizer._add_tokens(['[CLS]'])
    tokenizer._add_tokens(['[SEP]'])
    
    if not args.bart:
        tokenizer.bos_id = tokenizer.convert_tokens_to_ids('<s>')
        tokenizer.eos_id = tokenizer.convert_tokens_to_ids('</s>')

    model_config = AutoConfig.from_pretrained(args.pretrained_model)
    # model_config.decoder_start_token_id = 0
    model_config.vocab_size = len(tokenizer)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.kl_threshold = args.kl_threshold
    model_config.is_cvae = (args.dataset_type == 'wp')
    model_config.use_bow = args.use_bow
    model_config.begin_layer = args.begin_layer
    model_config.end_layer = args.end_layer
    model_config.diffusion_prior = args.diffusion_prior
    if args.diffusion_prior:
        model_config.sde_type = args.sde_type
        model_config.beta_end = args.beta_end
        model_config.beta_start = args.beta_start
        model_config.sigma2_0 = args.sigma2_0
        model_config.time_eps = args.time_eps
        model_config.diffusion_steps = args.diffusion_steps
        model_config.learn_sigma = args.learn_sigma
        model_config.sigma_small = args.sigma_small
        model_config.noise_schedule = args.noise_schedule
        model_config.use_kl = args.use_kl
        model_config.use_ddim = args.use_ddim
        model_config.clip_denoised = args.clip_denoised
        model_config.w = args.w
        model_config.predict_xstart = args.predict_xstart
        model_config.rescale_timesteps = args.rescale_timesteps
        model_config.rescale_learned_sigmas = args.rescale_learned_sigmas
        model_config.timestep_respacing = args.timestep_respacing
        model_config.schedule_sampler = args.schedule_sampler

    for arg in vars(args):
        if arg.startswith('latent'):
            setattr(model_config, arg, getattr(args, arg))
    
    model = Della(model_config) if not args.bart else BartDella(model_config)
    pretrained_model = AutoModel.from_pretrained(args.pretrained_model)
    logging.info('loading pretrained model parameters...')
    if not args.bart:
        init_para_frompretrained(model, pretrained_model)
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.wte = model.encoder.wte
    else:
        init_para_frompretrained_bart(model, pretrained_model)
        model.resize_token_embeddings(len(tokenizer))
        # model.decoder.wte = model.encoder.wte
    if args.load_epoch is not None:
        model_path = os.path.join(args.output_dir, args.model_name, 'model_epoch_{}.pt'.format(args.load_epoch))
        model_state_dict = torch.load(model_path, map_location=args.device)
        model.load_state_dict(model_state_dict)
        logging.info('load model_epoch_{}.pt finish'.format(args.load_epoch))
    else:
        args.load_epoch = -1

    if args.model_parallel and torch.cuda.device_count() > 1:  
        logging.info('model paralleize...')
        model.parallelize()
    else:
        model = model.to(args.device)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
    return model, tokenizer

def prepare_data(tokenizer, args):
    dataset_class = {'vae': VAEDataset, 'wp': WPDataset}
    if args.eval or args.generation:
        logging.info("eval model: the epoch {} of {}".format(args.load_epoch, args.model_name))
        test_dataset = dataset_class[args.dataset_type](args.test_file, tokenizer, args.device, bart=args.bart)
        test_iter = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        return test_iter
    else:
        train_dataset = dataset_class[args.dataset_type](args.train_file, tokenizer, args.device, bart=args.bart)
        valid_dataset = dataset_class[args.dataset_type](args.valid_file, tokenizer, args.device, bart=args.bart)
        train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        valid_iter = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn)
        logging.info('training with {} samples...'.format(len(train_dataset)))
        return train_iter, valid_iter

def main():
    args = get_args()
    prepare(args)
    model, tokenizer = prepare_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info('total parameters: {}'.format(total_params))
    if args.eval or args.generation:
        test_iter = prepare_data(tokenizer, args)
        if args.eval:
            valid(model, test_iter, args.load_epoch, args)
        if args.generation:
            generate(model, test_iter, tokenizer, args)
    else:
        train_iter, valid_iter = prepare_data(tokenizer, args)
        train(model, train_iter, valid_iter, args)

if __name__ == "__main__":
    main()

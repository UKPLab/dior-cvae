import torch
import os
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import logging
from tqdm import tqdm
from train_utils import *
from tensorboardX import SummaryWriter
import pandas as pd
logger = logging.getLogger(__name__)

def prepare_for_training(args, model, train_iter):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=True)
    t_total = len(train_iter) * args.epochs
    if args.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = None

    return model, optimizer, scheduler

def compute_loss(logits, target_tokens, kl_loss=None, beta=None, ignore_index=50256):
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_tokens[..., 1:].contiguous()
    
    ce_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    if kl_loss is not None:
        loss = ce_loss + beta * kl_loss
    else:
        loss = ce_loss
    return loss, ce_loss, kl_loss

def train(model, train_iter, valid_iter, args):
    logging.info('begin trainging...')
    model, optimizer, scheduler = prepare_for_training(args, model, train_iter)
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.model_name))
    if args.cycle_annealing:
        beta = 1e-5
        beta_0 = 1e-5
    else:
        beta = 1
    global_step = 0
    
    one_epoch_step = len(train_iter) // args.gradient_accumulation_steps
    beta_zero = beta_increase = args.cycle_iters // 2
    running_loss = 0
    running_ce_loss = 0
    running_kl_loss = 0
    running_bow_loss = 0
    for epoch in range(1 + args.load_epoch, args.epochs + args.load_epoch + 1):
        model.train()
        for i, inputs in enumerate(train_iter):
            model_output = model.train_model(**inputs) if args.bart else model(**inputs)
            if args.use_bow:
                ce_loss, kl_loss, bow_loss, _, _, ppl = model_output
                loss = ce_loss + beta * kl_loss + args.bow_weight * bow_loss
            else:
                ce_loss, kl_loss, _, _, ppl = model_output
                loss = ce_loss + beta * kl_loss
                    
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss = loss.mean()
            loss.backward()
            
            running_loss += loss.item()
            running_ce_loss += ce_loss.mean().item() / args.gradient_accumulation_steps
            running_kl_loss += kl_loss.mean().item() / args.gradient_accumulation_steps
            if args.use_bow:
                running_bow_loss += bow_loss.mean().item() / args.gradient_accumulation_steps
            
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                if args.cycle_annealing:
                    one_period = epoch % args.cycle_iters
                    if one_period < beta_zero:
                        beta = beta_0
                    else:
                        beta = min(1.0, beta + (1 - beta_0) / (beta_increase * one_epoch_step / 2))

                if global_step % args.log_step == 0:
                    logging.info('training loss: step [{}~{}], loss {}, ce_loss {}, kl_loss {}, bow_loss {}, lr {}, beta {}'.
                        format(global_step - args.log_step, global_step, running_loss / args.log_step, running_ce_loss / args.log_step, 
                                running_kl_loss / args.log_step, running_bow_loss / args.log_step, optimizer.param_groups[0]['lr'], beta))
                    writer.add_scalar('/train_loss', running_loss / args.log_step, global_step)
                    writer.add_scalar('/train_ce_loss', running_ce_loss / args.log_step, global_step)
                    writer.add_scalar('/train_kl_loss', running_kl_loss / args.log_step, global_step)
                    writer.add_scalar('/train_bow_loss', running_bow_loss / args.log_step, global_step)
                    writer.add_scalar('/lr', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('/beta', beta, global_step)
                    running_loss = 0
                    running_kl_loss = 0
                    running_ce_loss = 0
                    running_bow_loss = 0
        valid(model, valid_iter, epoch, args, writer, beta)
        save(model, args, epoch)
    logging.info('training finished')

def valid(model, valid_iter, epoch, args, tb_writer, beta=1):
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        valid_kl_loss = 0
        valid_ce_loss = 0
        valid_bow_loss = 0
        valid_ppl = 0
        valid_ppl_list = []
        for inputs in tqdm(valid_iter, desc='valid epoch {}'.format(epoch)):
            model_output = model.train_model(**inputs) if args.bart else model(**inputs)
            if args.use_bow:
                ce_loss, kl_loss, bow_loss, _, _, ppl = model_output
                loss = ce_loss + beta * kl_loss + args.bow_weight * bow_loss
            else:
                ce_loss, kl_loss, _, _, ppl = model_output
                loss = ce_loss + beta * kl_loss
            loss = loss.mean()
            valid_loss += loss.item()
            valid_ce_loss += ce_loss.mean().item()
            valid_kl_loss += kl_loss.mean().item()
            valid_ppl = valid_ppl + ppl
            valid_ppl_list.append(ppl)
            if args.use_bow:
                valid_bow_loss += bow_loss.mean().item()
        
        valid_loss = valid_loss / len(valid_iter)
        valid_ce_loss = valid_ce_loss / len(valid_iter)
        valid_kl_loss = valid_kl_loss / len(valid_iter)
        valid_bow_loss = valid_bow_loss / len(valid_iter)
        valid_ppl = valid_ppl / len(valid_iter)
        for p in valid_ppl_list:
            print()
        logging.info('valid result: epoch {}, loss {}, ce_loss {}, kl {}, bow {}, ppl {}'.format(epoch, valid_loss, valid_ce_loss, valid_kl_loss, valid_bow_loss, valid_ppl))
        if tb_writer is not None:
            tb_writer.add_scalar('/valid_loss', valid_loss, epoch)
            tb_writer.add_scalar('/valid_ce_loss', valid_ce_loss, epoch)
            tb_writer.add_scalar('/valid_kl_loss', valid_kl_loss, epoch)
            tb_writer.add_scalar('/valid_bow_loss', valid_bow_loss, epoch)

        if args.eval_metrics:
            ppl, elbo, nll, kl = calc_iwnll(model, valid_iter, ns=args.sample_times)
            #au = calc_au(model, valid_iter)
            logging.info('valid result: epoch {}, ppl {}, elbo {}, nll {}, kl {}'.format(epoch, ppl, elbo, nll, kl))
            #logging.info('valid result: epoch {}, au {}'.format(epoch, au))

def save(model, args, epoch):
    save_path = os.path.join(args.output_dir, args.model_name, 'model_epoch_{}.pt'.format(epoch))
    if not os.path.exists(os.path.join(args.output_dir, args.model_name)):
        os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    try:
        model_to_save = model.module
    except:
        model_to_save = model
    torch.save(model_to_save.state_dict(), save_path)

def generate(model, test_iter, tokenizer, args): 
    if args.dataset_type == 'wp':
        has_condition = "conditional"
    else:
        has_condition = "unconditional"
    if args.top_k > 0:
        generate_param = "topk_{}".format(args.top_k)
    elif args.greedy_decoding:
        generate_param = "greedy_decoding"
    else:
        generate_param = "beamsearch_{}".format(args.num_beams)
    
    logging.info('{} generate with {}'.format(has_condition, generate_param))
    def filter_sen(sen):
        sen = sen.replace('<sep>', '')
        sen = sen.replace('<s>', '')
        sen = sen.replace('</s>', '')
        sen = sen.replace('<pad>', '')
        sen = sen.replace('<|endoftext|>', '')
        sen = sen.replace('<eos>', '')
        sen = ' '.join(sen.split())
        return sen
    model.eval()
    model.decoder.config.is_encoder_decoder = False if not args.bart else True

    output_list = []
    target_list = []
    source_list = []
    
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(test_iter)):
            target = inputs['input_ids']
            if args.dataset_type == 'wp':
                source = inputs['condition']
            
            batch_size = target.size(0)
            device = target.device
            input_ids = target[:, 0].unsqueeze(1)
            model_kwargs = {}
            if args.dataset_type == 'wp':
                if args.visualize_prior:
                    latent_dir = os.path.join(args.latent_dir, args.model_name)
                    if not os.path.exists(latent_dir):
                        os.makedirs(latent_dir, exist_ok=True)
                    prior_latent_list_batch = []
                    for i in range(5000):
                        prior_latent, encoder_outputs = model.get_prior(batch_size, device, condition=inputs['condition'], condition_mask=inputs['condition_mask'])
                        prior_latent_list_batch.append(torch.flatten(torch.stack(prior_latent).permute(1, 0, 2), start_dim=1, end_dim=-1).cpu().split(1, 0))
                    for i in range(prior_latent[0].shape[0]):
                        single_list = []
                        for j in range(5000):
                            single_list.append(prior_latent_list_batch[j][i].squeeze(0).numpy())
                        single_df = pd.DataFrame(single_list)
                        single_df.to_csv(latent_dir + '/' + str(idx * prior_latent[0].shape[0] + i) + '.csv', header=None, index=None)
                    continue
                else:
                    prior_latent, encoder_outputs = model.get_prior(batch_size, device, condition=inputs['condition'], condition_mask=inputs['condition_mask'])
                model_kwargs['attention_mask'] = inputs['condition_mask']
                model_kwargs['encoder_outputs'] = encoder_outputs
                input_ids = inputs['condition']
                if args.bart:
                    model_kwargs['compute_kl'] = False
                    model_kwargs['output_attentions'] = True
                    model_kwargs['output_hidden_states'] = True
            else:
                prior_latent = model.get_prior(batch_size, device)

            gen_model = model.decoder if not args.bart else model
            if args.top_k > 0:
                ans = gen_model.generate(
                    input_ids, 
                    latent=prior_latent, 
                    bos_token_id=tokenizer.bos_id if not args.bart else tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_id if not args.bart else tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_id if not args.bart else tokenizer.pad_token_id,
                    do_sample=True,
                    top_k=args.top_k, 
                    top_p=args.top_p, 
                    min_length=input_ids.size(-1) + 3 if not args.bart else 3,
                    max_length=min(args.max_length, 1024),
                    repetition_penalty=args.repetition_penalty, 
                    **model_kwargs,
                )
            elif args.greedy_decoding:
                ans = gen_model.generate(
                    input_ids, 
                    latent=prior_latent, 
                    bos_token_id=tokenizer.bos_id if not args.bart else tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_id if not args.bart else tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_id if not args.bart else tokenizer.pad_token_id,
                    min_length=input_ids.size(-1) + 3 if not args.bart else 3,
                    max_length=min(args.max_length, 1024),
                    repetition_penalty=args.repetition_penalty, 
                    **model_kwargs,
                )
            else:
                if prior_latent is not None:
                    if isinstance(prior_latent, tuple):
                        latent = [item.repeat_interleave(args.num_beams, dim=0) for item in prior_latent]
                    else:
                        latent = prior_latent.repeat_interleave(args.num_beams, dim=0)
                else:
                    latent = None
                ans = gen_model.generate(
                    input_ids,
                    latent=latent, 
                    bos_token_id=tokenizer.bos_id if not args.bart else tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_id if not args.bart else tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_id if not args.bart else tokenizer.pad_token_id,
                    num_beams=args.num_beams, 
                    min_length=input_ids.size(-1) + 3 if not args.bart else 3,
                    max_length=min(args.max_length, 1024), 
                    repetition_penalty=args.repetition_penalty, 
                    **model_kwargs,
                )
            ans = ans.cpu().numpy()

            if args.dataset_type == 'wp':
                target = target.cpu().numpy()
                source = source.cpu().numpy()
            for i in range(len(ans)):
                text_ans = tokenizer.decode(ans[i], clean_up_tokenization_spaces=False)
                text_ans = filter_sen(text_ans)
                if len(text_ans) > 0:
                    output_list.append(text_ans)
                    if args.dataset_type in 'wp':
                        target_text = tokenizer.decode(target[i], clean_up_tokenization_spaces=False)
                        target_text = filter_sen(target_text)
                        target_list.append(target_text)
                        source_text = tokenizer.decode(source[i], clean_up_tokenization_spaces=False)
                        source_text = filter_sen(source_text)
                        source_list.append(source_text)
                else:
                    print('meet zero')

    save_dir = os.path.join(args.generation_output_dir, args.model_name)
    file_name = '{}_output_{}_epoch_{}_outputs.txt'.format(has_condition, generate_param, args.load_epoch)
    logging.info('generation output save at {}'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, file_name), 'w') as f:
        f.write('\n'.join(output_list))
    if args.dataset_type == 'wp':
        file_name = '{}_output_{}_epoch_{}_targets.txt'.format(has_condition, generate_param, args.load_epoch)
        with open(os.path.join(save_dir, file_name), 'w') as f:
            f.write('\n'.join(target_list))
        file_name = '{}_output_{}_epoch_{}_sources.txt'.format(has_condition, generate_param, args.load_epoch)
        with open(os.path.join(save_dir, file_name), 'w') as f:
            f.write('\n'.join(source_list))

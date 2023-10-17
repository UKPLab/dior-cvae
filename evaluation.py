from collections import Counter
import sys
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import os
from tensorboardX import SummaryWriter
from evaluate import load

import argparse

def bleu(hyps: list, refs: list, ignore_indices=None, output_path=None) -> tuple:
    # hyps, refs = load_file(hyps_file_path, refs_file_path, ignore_indices)
    bleu_1, bleu_2 = [], []
    for hyp, ref in zip(hyps, refs):
        hyp = hyp.strip().split()
        ref = ref.strip().split()
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp, smoothing_function=SmoothingFunction().method7, weights=[1, 0, 0, 0])
        except Exception as e:
            print(e)
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp, smoothing_function=SmoothingFunction().method7, weights=[0.5, 0.5, 0, 0])
        except Exception as e:
            print(e)
            score = 0
        bleu_2.append(score)
    bleu_1, bleu_2 = np.average(bleu_1), np.average(bleu_2)
    output_content = 'BLEU-1/2: {}/{}\n'.format(round(bleu_1, 4), round(bleu_2, 4))
    print('-------------- BLEU score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return bleu_1, bleu_2


# calculate Distinct-1/2 for dailydialog & personachat
def distinct(hyps: list, output_path=None) -> tuple:
    # hyps, _ = load_file(hyps_file_path)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for hyp in hyps:
        hyp = hyp.strip().split()
        unigrams = Counter(hyp)
        bigrams = Counter(zip(hyp, hyp[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(hyp)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(hyp)-1)+1e-5))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)
    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    output_content = 'Distinct-1/2: {}/{}\n'.format(round(inter_dist1, 4), round(inter_dist2, 4))
    print('-------------- Distinct score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def sort_result(ex):
    return ex['inter_dist1'] + ex['inter_dist2'] + ex['bleu_1'] + ex['bleu_2']
    # return ex['inter_dist1'] + ex['inter_dist2']

def name_sort(name):
    line = name.rfind('_')
    return int(name[line + 1:-4])


prefix = './generation_output/della-bart-diffusion-memdrop05start-bos_withcycleann_withconstant_dd/conditional_output_beamsearch_10_epoch_39_'
out_prefix = './generation_output/llm/dailydialog_llama-2-hf.txt'
out_path = out_prefix
# prefix = './data/daily_dialog/valid_results/variational-ckpts_lr2_batch2048_warm20000_full-vocab_share-enc_ncvae_bart_latent-64_diff_enc-drop04-bos_newdata/'
# ref_path = './data/personachat/processed_optimus_knowledge_madeupword/test.tgt'
# ref_path = prefix[:-14] + 'targets.txt'
ref_path = prefix + 'targets.txt'
# dd_filter_idx = np.load('./data/daily_dialog/filter_idx.npy').tolist()
result = []
with open(ref_path) as f:
    gold = f.readlines()
with open(out_path) as f:
    generated = f.readlines()
final_gold = []
final_generated = []
loop = zip(gold, generated)
for i, (gld, gen) in enumerate(loop):
    # if i in dd_filter_idx:
    #     continue
    # final_gold.append(gld.strip().replace('...', ' ...').replace(' ,', ',').replace(' .', '.').replace(' ’', '’').replace(' ?', '?').replace(' !', '!').replace(' \'', '\'').replace('$ ', '$').replace(' ;', ';').replace(' :', ':').replace(' (', '(').replace(' )', ')'))
    # final_generated.append(gen.strip().replace(' ,', ',').replace('...', ' ...').replace(' .', '.').replace(' ’', '’').replace(' ?', '?').replace(' !', '!').replace(' \'', '\'').replace('$ ', '$').replace(' ;', ';').replace(' :', ':').replace(' (', '(').replace(' )', ')'))
    final_gold.append(gld.strip())
    final_generated.append(gen.strip())


intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(final_generated)
bleu_1, bleu_2 = bleu(final_generated, final_gold)
bert_score_metric = load("bertscore")
bert_score_result = bert_score_metric.compute(predictions=final_generated, references=final_gold, lang="en")
bert_score_mean = {}
bert_score_mean['precision'] = sum(bert_score_result['precision']) / len(bert_score_result['precision'])
bert_score_mean['recall'] = sum(bert_score_result['recall']) / len(bert_score_result['recall'])
bert_score_mean['f1'] = sum(bert_score_result['f1']) / len(bert_score_result['f1'])
eval_results = {}
eval_results['intra_dist1'] = intra_dist1
eval_results['intra_dist2'] = intra_dist2
eval_results['inter_dist1'] = inter_dist1
eval_results['inter_dist2'] = inter_dist2
eval_results['bleu_1'] = bleu_1
eval_results['bleu_2'] = bleu_2
with open(prefix + 'evaluation_results_all.txt', 'w') as writer:
    for key in sorted(eval_results.keys()):
        print("%s = %s", key, str(eval_results[key]))
        writer.write("%s = %s\n" % (key, str(eval_results[key])))

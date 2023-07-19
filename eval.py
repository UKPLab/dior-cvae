from collections import Counter
import sys
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import os


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

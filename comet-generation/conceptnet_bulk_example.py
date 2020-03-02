import os
import sys
import argparse
import torch
import json 
import tqdm
import nltk
from nltk import ngrams
sys.path.append(os.getcwd())
import tokenization 
tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

with open('stopwords.json', 'r') as f:
    stopwords = json.load(f)
exclude_words = ['I', '\'', 'm', 's', 'am']
def get_ngrams(tokens):
    bigrams = []
    for gram in ngrams(tokens, 2):
        if all([w in stopwords for w in gram]):
            continue
        if not all([w not in exclude_words for w in gram]):
            continue
        bigrams.append(' '.join(gram))
    trigrams = []
    for gram in ngrams(tokens, 3):
        if all([w in stopwords for w in gram]):
            continue
        if not all([w not in exclude_words for w in gram]):
            continue
        trigrams.append(' '.join(gram))
    return trigrams + bigrams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_15500.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")
    parser.add_argument("--input_data", type=str, default=None)
    parser.add_argument("--outname", type=str, default=None)

    args = parser.parse_args()
    with open(args.input_data, 'r') as f:
        raw_data = json.load(f)

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    sampling_algorithm = args.sampling_algorithm
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
    relation = [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
    'CreatedBy', 'DesireOf', 'Desires',
    'HasFirstSubevent', 'HasLastSubevent', 'HasPrerequisite', 'HasProperty',
    'HasSubevent', 'InheritsFrom', 'IsA',
    'LocatedNear', 'LocationOfAction', 'MotivatedByGoal',
    'PartOf', 'ReceivesAction', 'RelatedTo',
    'SymbolOf', 'UsedFor'
    ]   
    results_cache = {}
    for sample in tqdm.tqdm(raw_data):
        for turn in sample[0]:
            content = turn.split(':')[1].strip()
            sents = nltk.sent_tokenize(content)
            for sent in sents:
                sent = sent.strip()[:-1]
                subsents = sent.split(',')
                for subsent in subsents:
                    tokens = tokenizer.tokenize(subsent)
                    candidates = get_ngrams(tokens)
                    for cand in candidates:
                        if cand not in results_cache:
                            outputs = interactive.get_conceptnet_sequence(cand, model, sampler, data_loader, text_encoder, relation)
                            results_cache[cand] = outputs
        for qa in sample[1]:
            ques = qa['question'][:-1]
            subqs = qa['question'].split(',')
            for subq in subqs:
                subq = subq.strip()
                tokens = tokenizer.tokenize(subq)
                candidates = get_ngrams(tokens)
                for cand in candidates:
                    if cand not in results_cache:
                        outputs = interactive.get_conceptnet_sequence(cand, model, sampler, data_loader, text_encoder, relation)
                        results_cache[cand] = outputs
            for option in qa['choice']:
                option = option[:-1]
                subos = option.split(',')
                for subo in subos:
                    subo = subo.strip()
                    tokens = tokenizer.tokenize(subo)
                    candidates = get_ngrams(tokens)
                    for cand in candidates:
                        if cand not in results_cache:
                            outputs = interactive.get_conceptnet_sequence(cand, model, sampler, data_loader, text_encoder, relation)
                            results_cache[cand] = outputs
    with open(args.outname, 'w') as fout:
        json.dump(results_cache, fout)



    


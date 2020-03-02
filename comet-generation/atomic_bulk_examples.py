import os
import sys
import argparse
import torch
import tqdm
import json
import nltk
import spacy
nlp = spacy.load('en_core_web_sm')
sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import myfunctions as interactive

def pretty_print(sample):
    for turn in sample[0]:
        print (turn)
    print ()
    for turn in sample[3]:
        for atom in turn:
            atom_list = []
            for k, v in atom.items():
                if v['beams'][0] != 'none':
                    atom_list.append((k, v['beams'][0]))
            print (atom_list)
    for qa in sample[1]:
        print (qa['question'])
        for atom in qa['question_atomic']:
            atom_list = []
            for k, v in atom.items():
                if v['beams'][0] != 'none':
                    atom_list.append((k, v['beams'][0]))
            print (atom_list)
        print ()
        for i, option in enumerate(qa['choice']):
            print (option)
            atom_list = []
            for atom in qa['option_atomic'][i]:
                for k, v in atom.items():
                    if v['beams'][0] != 'none':
                        atom_list.append((k, v['beams'][0]))
            print (atom_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")
    parser.add_argument("--input_data", type=str, default=None)
    parser.add_argument("--outname", type=str, default=None)

    args = parser.parse_args()
    with open(args.input_data, 'r') as f:
        raw_data = json.load(f)

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
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
    category = "all"
    for sample in tqdm.tqdm(raw_data):
        dialog_atomic = []
        for turn in sample[0]:
            turn_atomic = []
            content = turn.split(':')[1].strip()
            sents = nltk.sent_tokenize(content)
            for sent in sents:
                sent = sent.strip()
                #parsed_sent = nlp(sent)
                #pos = [w.tag_[:2] for w in parsed_sent]
                #if 'VB' not in pos:
                #    continue
                prefix_sent, _ = data.atomic_data.do_example(text_encoder, sent, None, True, None)
                if len(prefix_sent) > 17:
                    subsents = sent.split(',')
                    for subsent in subsents:
                        subsent = subsent.strip()
                        prefix_subsent, _ = data.atomic_data.do_example(text_encoder, subsent, None, True, None)
                        prefix_subsent = prefix_subsent[:17]
                        outputs_subsent = interactive.get_atomic_sequence(prefix_subsent, model, sampler, data_loader, text_encoder, category)
                        turn_atomic.append(outputs_subsent)
                else:
                    outputs_sent = interactive.get_atomic_sequence(prefix_sent, model, sampler, data_loader, text_encoder, category)
                    turn_atomic.append(outputs_sent)
            dialog_atomic.append(turn_atomic)
        sample.append(dialog_atomic)
        for qa in sample[1]:
            qa['question_atomic'] = []
            #parsed_question = nlp(qa['question'])
            #pos = [w.tag_[:2] for w in parsed_question]
            #if 'VB' in pos:
            prefix_q, _ = data.atomic_data.do_example(text_encoder, qa['question'], None, True, None)
            if len(prefix_q) > 17:
                subqs = qa['question'].split(',')
                for subq in subqs:
                    subq = subq.strip()
                    prefix_subq, _ = data.atomic_data.do_example(text_encoder, subq, None, True, None)
                    prefix_subq = prefix_subq[:17]
                    outputs_subq = interactive.get_atomic_sequence(prefix_subq, model, sampler, data_loader, text_encoder, category)
                    qa['question_atomic'].append(outputs_subq)
            else:
                outputs_q = interactive.get_atomic_sequence(prefix_q, model, sampler, data_loader, text_encoder, category)
                qa['question_atomic'].append(outputs_q)
            qa['option_atomic'] = []
            for option in qa['choice']:
                o_atomic = []
                #parsed_option = nlp(option)
                #pos = [w.tag_[:2] for w in parsed_option]
                #if 'VB' not in pos:
                #    qa['option_atomic'].append(o_atomic)
                #    continue
                #else:
                prefix_o, _ = data.atomic_data.do_example(text_encoder, option, None, True, None)
                #print (option)
                #print (prefix_o)
                if len(prefix_o) > 17:
                    subos = option.split(',')
                    for subo in subos:
                        subo = subo.strip()
                        prefix_subo, _ = data.atomic_data.do_example(text_encoder, subo, None, True, None)
                        prefix_subo = prefix_subo[:17]
                        outputs_subo = interactive.get_atomic_sequence(prefix_subo, model, sampler, data_loader, text_encoder, category)
                        o_atomic.append(outputs_subo)
                else:
                    outputs_o = interactive.get_atomic_sequence(prefix_o, model, sampler, data_loader, text_encoder, category)
                    o_atomic.append(outputs_o)
                qa['option_atomic'].append(o_atomic)
        #pretty_print(sample)
        #exit(0)
    with open(args.outname, 'w') as fout:
        json.dump(raw_data, fout)



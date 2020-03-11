# Towards Generalizable Neuro-Symbolic Systems for Commonsense Question Answering
This repository contains the code for the paper "Towards Generalizable Neuro-Symbolic Systems for Commonsense
Question Answering". See full paper [here](https://arxiv.org/abs/1910.14087)

## Enviroments
This code has been tested on Python 3.6.7, Pytorch 1.2 and Transformers 2.3.0 

## Dataset
Please downdload the CommonsenseQA dataset from [the official website](https://www.tau-nlp.org/commonsenseqa)

## ConceptNet Knowledge Extraction
Go to directory Extraction, then download ConceptNet from [the official website](https://github.com/commonsense/conceptnet5/wiki/Downloads) and uncompress in the current directory. 
Then simple run:
```
python extract_english.py 
python extract4commonsenseqa.py
```
This would generate the files in the data directory. 
To run extraction on datasets other than CommonsenseQA, you will need to modify the data loading and formatting accordingly. 

## ATOMIC Knowledge Generation 
If you only need to train models on CommonsenseQA, you can skip this part. 
First clone the [comet official repo](https://github.com/atcbosselut/comet-commonsense) and put files in comet-generation under scripts/interactive 
Then follow the instructions from official repo to download necessary data/models to run generation

## HyKAS Model Training
```
CUDA_VISIBLE_DEVICES=0 python run_csqa.py --data_dir data/ --model_type roberta-ocn-inj --model_name_or_path 
roberta-large --task_name csqa-inj --cache_dir downloaded_models --max_seq_length 80 --do_train --do_eval 
--evaluate_during_training --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-5 
--num_train_epochs 8 --warmup_steps 150 --output_dir workspace/hykas
```
For CommonsenseQA, the Dev accuracy should get around 79%. 

## Cite 
```
@inproceedings{ma-etal-2019-towards,
    title = "Towards Generalizable Neuro-Symbolic Systems for Commonsense Question Answering",
    author = "Ma, Kaixin and Francis, Jonathan and Lu, Quanyang and Nyberg, Eric and Oltramari, Alessandro",
    booktitle = "Proceedings of the First Workshop on Commonsense Inference in Natural Language Processing",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-6003",
    doi = "10.18653/v1/D19-6003",
    pages = "22--32",
}
```

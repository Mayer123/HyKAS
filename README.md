# ConceptNet Knowledge Extraction
Go to directory Extraction, then download ConceptNet from [the official website](https://github.com/commonsense/conceptnet5/wiki/Downloads) and uncompress in the current directory. 
Then simple run:
```
python extract_english.py 
python extract4commonsenseqa.py
```
## Note
To run this extraction on datasets other than CommonsenseQA, you will need to modify the data loading and formatting accordingly. 

# ATOMIC Knowledge Generation 
First clone the [comet official repo](https://github.com/atcbosselut/comet-commonsense) and put files in comet-generation under scripts/interactive 
Then follow the instructions from official repo to download necessary data/models to run generation

# HyKAS Model Training
```
CUDA_VISIBLE_DEVICES=0 python run_csqa.py --data_dir CommonsenseQA --model_type roberta-ocn-inj --model_name_or_path roberta-large --task_name csqa-inj --cache_dir downloaded_models --max_seq_length 80 --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-5 --num_train_epochs 8 --warmup_steps 150 --output_dir workspace/hykas
```

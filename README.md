# SentiPrompt

This is the official implementation of the paper [SentiPrompt: Sentiment Knowledge Enhanced Prompt-Tuning for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2109.08306)

## Dependency
Install the package in the requirements.txt, then use the following
commands to install two other packages
```text
pip install git+https://github.com/fastnlp/fastNLP@dev
pip install git+https://github.com/fastnlp/fitlog
```
## Dataset Preparation
All versions of datasets should be placed in data/ folder and different datasets split in sub-directories.
```text
 -  data
    - penga  # D_20a in paper
    - pengb  # D_20b in paper
    - lcx    # D_21 in paper  
```
and data for train, dev and test should be placed inside every sub-directories in json format.
## Training
Run train.py to start a training. Use --dataset_name to assign the dataset you want to train with, --prompt to choose the prompt type and --fewshot for fewshot setting. For example
```text
LOG_DIR=logs_prompt_type3/pengb/
SEED=your_seed
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_name pengb/14lap \
                                       --log_dir $LOG_DIR/14lap \
                                       --batch_size 16 \
                                       --lr 5e-5 \
                                       --n_epochs 100 \
                                       --save_model $LOG_DIR/14lap/models \
                                       --prompt type3 \
                                       --seed $SEED \
                                       --fewshot
```


Please do remember to cite this paper if you use our published dataset.

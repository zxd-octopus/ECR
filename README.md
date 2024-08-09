This is the implementation for the Recsys 2024 paper:

Towards Empathetic Conversational Recommender System

## Requirements

- python == 3.8.13
- pytorch == 1.8.1
- cudatoolkit == 11.1.1
- transformers == 4.15.0
- pyg == 2.0.1
- accelerate == 0.8.0

## Data

The data we used has been uploaded [here]().

The downloaded ckpt files should be moved into src_emo/data/emo_data/.

## Saved Models

We have saved the parameters of our model, all of which have been uploaded [here]().

The downloaded ckpt files should be moved into src_emo/data/saved/.

## Quick-Start

We run all experiments and tune hyperparameters on a GPU with 24GB memory, you can adjust `per_device_train_batch_size` and `per_device_eval_batch_size` according to your GPU, and then the optimization hyperparameters (e.g., `learning_rate`) may also need to be tuned.

### Emotional Semantic Fusion Subtask

```bash
cd src_emo 
cp -r data/emo_data/* data/redial/
python data/redial/process.py 
accelerate launch train_pre.py \
--dataset redial \
--num_train_epochs 10 \
--gradient_accumulation_steps 4  \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 64  \
--num_warmup_steps 1389 \
--max_length 200 \
--prompt_max_length 200  \
--entity_max_length 32  \
--learning_rate 5e-4 \
--seed 42 \
--nei_mer

```

### Emotion-aware Item Recommendation Subtask Training

```bash
# merge infer results from conversation model of UniCRS
cd src_emo
cp -r data/emo_data/* data/redial/
python data/redial/process_mask.py
cp -r data/redial/* data/redial_gen/
python data/redial_gen/merge.py
accelerate launch train_rec.py   \
--dataset redial_gen   \
--n_prefix_rec 10    \
--num_train_epochs 5   \
--per_device_train_batch_size 16   \
--per_device_eval_batch_size 32   \
--gradient_accumulation_steps 8   \
--num_warmup_steps 530   \
--context_max_length 200   \
--prompt_max_length 200   \
--entity_max_length 32   \
--learning_rate 1e-4   \
--seed 8   \
--like_score 2.0   \
--dislike_score 1.0   \
--notsay_score 0.5    \
--weighted_loss   \
--nei_mer  \
--use_sentiment 

```

### Emotion-aligned Response Generation Training and Inference

#### Backbone: DialoGPT

```bash
# merge infer results from Recommend Response Generation
# train
cd src_emo
python data/redial_gen/merge_rec.py
python imdb_review_entity_filter.py
python data/redial/process_empthetic.py
accelerate launch train_emp.py   \
 --dataset redial   \
--num_train_epochs 15    \
--gradient_accumulation_steps 1    \
--ignore_pad_token_for_loss    \
--per_device_train_batch_size 20    \
--per_device_eval_batch_size 64    \
--num_warmup_steps 9965    \
--context_max_length 150    \
--resp_max_length 150    \
--learning_rate 1e-04  

# infer
accelerate launch infer_emp.py   \
--dataset redial_gen \
--split test \
--per_device_eval_batch_size 256 \
--context_max_length 150 \
--resp_max_length 150
```

#### Backbone: Llama 2-Chat
```
We use [LLaMA Board](https://github.com/hiyouga/LLaMA-Efficient-Tuning) to fine-tune  Llama 2-Chat.

Training Data Path: src_emo/emo_data/llama_train.json

Testing Data Path: src_emo/emo_data/llama_test.json
```

## Acknowledgement

Our datasets and data process code are developed based on [UniCRS](https://github.com/RUCAIBox/UniCRS).

Any scientific publications that use our codes or dataset should cite our paper as the reference.
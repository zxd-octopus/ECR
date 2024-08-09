import argparse
import os
import sys
import time

import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_emp import CRSEmpDataCollator, CRSEmpDataset
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_max_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--resp_max_length', type=int, help="max length of decoder input.")
    parser.add_argument("--tokenizer", type=str, default="save/dialogpt/")
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    parser.add_argument("--text_tokenizer", type=str, default="roberta-base")
    # model
    parser.add_argument("--model", type=str, default="data/saved/emp/")
    parser.add_argument("--max_gen_len", type=int, default=150)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--wk", action="store_true", default=False)
    parser.add_argument("--wt", action="store_true", default=False)
    parser.add_argument("--wn", action="store_true", default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    kg = DBpedia(dataset=args.dataset, debug=args.debug)

    # data
    dataset = CRSEmpDataset(
        args.dataset, args.split, tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,infer = True,kg = kg, sample = args.sample, wk = args.wk, wt = args.wt, wn = args.wn
    )
    data_collator_generator = CRSEmpDataCollator(
        tokenizer=tokenizer, device=device, gen=True, use_amp=accelerator.use_fp16, debug=args.debug,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )
    gen_dir = os.path.join('save', args.dataset)
    os.makedirs(gen_dir, exist_ok=True)
    model_name = args.model.split('/')[-2]
    if args.sample:
        args.split = 'sample'
    if args.wk:
        args.split = 'sample' + 'wk'
    if args.wt :
        args.split = 'sample' + 'wt'
    if args.wn:
        args.split = 'sample' + 'wn'
    gen_file_path = os.path.join(gen_dir, f'{model_name}_{args.split}.jsonl')

    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            gen_seqs = accelerator.unwrap_model(model).generate(
                **batch['context'],
                max_new_tokens=args.max_gen_len,
                no_repeat_ngram_size=3,
            )
            gen_resp_ids = []
            for gen_seq, length in zip(gen_seqs, batch['context_len']):
                gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq[length:])
            evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process, context = batch['context']['input_ids'])

    # metric
    accelerator.wait_for_everyone()
    report = evaluator.report()
    test_report = {}
    for k, v in report.items():
        test_report[f'{args.split}/{k}'] = v
    logger.info(test_report)
    if run:
        run.log(test_report)

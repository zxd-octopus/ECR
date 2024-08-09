import argparse
import math
import os
import sys
import time
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from dataset_dbpedia import DBpedia
from dataset_pre import CRSDataset, CRSDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict, Emo_List, seed_torch
from model_prompt import KGPrompt
from  co_appear import TOCoAppear


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='data/saved/pre-trained', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str,default='redial', required=False, help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str,default="save/dialogpt/")
    parser.add_argument("--text_tokenizer", type=str,default = "save/roberta/")
    # model
    parser.add_argument("--model", type=str, required=False,default = "save/dialogpt/",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str,default = "save/roberta/")
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    # optim
    parser.add_argument("--prompt_encoder", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--use_new_seed", action="store_true", default=False)
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    # addition for crs_emp
    parser.add_argument("--weighted_loss", action="store_true", default=False)
    parser.add_argument("--remove_neg", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--use_sentiment", action="store_true", default=False)
    parser.add_argument("--like_score", type=float, default=2.0)
    parser.add_argument("--dislike_score", type=float, default=0.3)
    parser.add_argument("--notsay_score", type=float, default=1.0)
    parser.add_argument('--copy', action='store_true', default=False)
    parser.add_argument('--copy_w', type=float, default=0.5)
    parser.add_argument("--RESUME", action="store_true", help="")
    parser.add_argument("--nei_mer", action="store_true", help="")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)


    if args.seed is not None:
        if args.use_new_seed:
            seed_torch(args.seed)
        else:
            set_seed(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(config)
    logger.info(accelerator.state)
    args.output_dir = args.output_dir + local_time

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

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg_class = DBpedia(dataset=args.dataset, debug=args.debug)
    kg = kg_class.get_entity_kg_info()
    toca = TOCoAppear(kg_class)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model) # todo read
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    emo2idx = {emo: id for id, emo in enumerate(Emo_List)}

    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_emotion = len(Emo_List)
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    fix_modules = [model, text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)

    # optim & amp
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # data
    train_dataset = CRSDataset(
        dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length, emotion_max_length = 3, emo2idx = emo2idx
        , kg=kg_class, toca=toca
    )
    valid_dataset = CRSDataset(
        dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length, emotion_max_length = 3, emo2idx = emo2idx, kg = kg_class, toca = toca
    )
    test_dataset = CRSDataset(
        dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length, emotion_max_length = 3, emo2idx = emo2idx, kg = kg_class, toca = toca
    )
    data_collator = CRSDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        max_length=args.max_length, entity_max_length=args.entity_max_length,
        use_amp=accelerator.use_fp16, debug=args.debug,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length, pad_emotion_id = len(Emo_List), emotion_max_length=3,n_entity = kg["num_entities"]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    evaluator = RecEvaluator()
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    start_epoch = 0

    # for nan loss detect
    if args.RESUME:
        path_checkpoint = "data/train_ckpt/ckpt_0.pth" 
        checkpoint = torch.load(path_checkpoint) 

        prompt_encoder.load_state_dict(checkpoint['net'])  

        optimizer.load_state_dict(checkpoint['optimizer'])  
        start_epoch = checkpoint['epoch'] 
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])  

    # train loop
    for epoch in range(start_epoch, args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()
        # train
        for step, batch in enumerate(train_dataloader):
            if step == 5908:
                print("here")
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds, copy_logit = prompt_encoder(
                entity_ids=batch['entity'],
                emotion_ids=batch['emotion'],
                token_embeds=token_embeds,
                output_entity=True,
                use_copy=args.copy,
                nei_mvs=batch['nei_mvs'],
                emotion_probs = batch['emotion_probs'],
                nei_mer = args.nei_mer
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            loss = model(**batch['context'], rec=True, copy_logit = copy_logit, copy_w = args.copy_w).rec_loss / args.gradient_accumulation_steps
            logger.info(loss)
            if loss.isnan():
                logger.info("***** loss is nan here *****")
            accelerator.backward(loss, retain_graph = True)
            train_loss.append(float(loss))

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        del train_loss, batch

        # for nan loss detect
        print('epoch:', epoch)
        print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
        checkpoint = {
            "net": prompt_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': lr_scheduler.state_dict()
        }
        if not os.path.isdir("data/train_ckpt"):
            os.mkdir("data/train_ckpt")
        torch.save(checkpoint, 'data/train_ckpt/ckpt_%s.pth' % (str(epoch)))

        # valid
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds, copy_logit = prompt_encoder(
                    entity_ids=batch['entity'],
                    emotion_ids=batch['emotion'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_copy=args.copy,
                    nei_mvs=batch['nei_mvs'],
                    emotion_probs = batch['emotion_probs'],
                nei_mer = args.nei_mer
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True, copy_logit = copy_logit, copy_w = args.copy_w)
                valid_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # metric
        report, report_add = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / report['count']
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test
        test_loss = []
        prompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds, copy_logit = prompt_encoder(
                    entity_ids=batch['entity'],
                    emotion_ids=batch['emotion'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_copy=args.copy,
                    nei_mvs=batch['nei_mvs'],
                    emotion_probs = batch['emotion_probs'],
                nei_mer = args.nei_mer
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True, copy_logit = copy_logit, copy_w = args.copy_w)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # metric
        report, report_add = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {}
        for k, v in report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v / report['count']
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

        # epoch_dir = os.path.join(args.output_dir, str(epoch))
        # os.makedirs(epoch_dir, exist_ok=True)
        # prompt_encoder.save(epoch_dir)
        # logger.info(f'save model of epoch {epoch}')

    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')

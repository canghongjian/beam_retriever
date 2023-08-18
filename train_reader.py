import logging
import os
import random
from datetime import date
import json
import collections
import re
import string

import numpy as np
import torch
import transformers
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from qa.reader_model import Reader
from qa.datasets import MHReaderDataset, reader_mhop_collate
from utils.utils import load_saved, move_to_cuda, AverageMeter
from qa.config import train_args

def main():
    args = train_args()
    transformers.logging.set_verbosity_error()
    if args.fp16:
        # import apex
        # apex.amp.register_half_function(torch, 'einsum')
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)
    tb_path = os.path.join(args.output_dir, "tblogs")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path, exist_ok=True)

    tb_logger = SummaryWriter(tb_path)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab                          
    
    bert_config = AutoConfig.from_pretrained(args.model_name)
    bert_config.max_position_embeddings = args.max_seq_len
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.dataset_type == 'hotpot':
        # hotpot format
        Sentence_token = "</e>"
        DOC_token = "</d>"
        tokenizer.add_tokens([Sentence_token, DOC_token])
    model = Reader(bert_config, args.model_name, task_type=args.dataset_type, len_tokenizer=len(tokenizer) if args.dataset_type == 'hotpot' else 0, gradient_checkpointing=args.gradient_checkpointing)
    eval_dataset = MHReaderDataset(
    tokenizer, args.predict_file, max_len=args.max_seq_len, type=args.dataset_type, is_train=False)

    eval_dataloader = DataLoader(
    eval_dataset, batch_size=args.predict_batch_size, pin_memory=True,
    num_workers=args.num_workers, collate_fn=reader_mhop_collate)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.do_train and args.max_seq_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_len, bert_config.max_position_embeddings))

    if args.local_rank == -1 or args.local_rank == 0:
        logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        if args.local_rank == -1 or args.local_rank == 0:
            logger.info(f"begin load trained model from :{args.init_checkpoint}")
        model = load_saved(model, args.init_checkpoint)

    model.to(device)

    if args.local_rank == -1 or args.local_rank == 0:
        logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters,
                         lr=args.learning_rate, eps=args.adam_epsilon)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    

    if args.do_train:
        global_step = 0  # gradient update step
        batch_step = 0  # forward batch count
        best_f1 = 0
        train_loss_meter = AverageMeter()
        model.train()

        train_dataset = MHReaderDataset(tokenizer, args.train_file, max_len=args.max_seq_len, type=args.dataset_type, is_train=True)
        if args.local_rank != -1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True,
                                        num_workers=args.num_workers, sampler=train_sampler, collate_fn=reader_mhop_collate)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True,
                                        num_workers=args.num_workers, shuffle=True, collate_fn=reader_mhop_collate)
    
        
        t_total = len(train_dataloader) // args.accumulate_gradients * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        log_steps = int(len(train_dataloader) // args.accumulate_gradients * args.log_period_ratio)
        eval_steps = int(len(train_dataloader) // args.accumulate_gradients * args.eval_period_ratio)

        if args.local_rank == -1 or args.local_rank == 0:
            logger.info(f'Start training.... log_steps:{log_steps}, eval_steps:{eval_steps}')
        for epoch in range(int(args.num_train_epochs)):
            for batch in tqdm(train_dataloader):
                batch_step += 1
                id = batch.pop('id')
                answer = batch.pop('answer')
                if 'sentence_num' in batch:
                    sentence_num = batch.pop('sentence_num')
                batch = move_to_cuda(batch)
                if args.fp16:
                    with autocast():
                        output = model(**batch)
                else:
                    output = model(**batch)
                loss = output['loss'].sum()
                if args.accumulate_gradients > 1:
                    loss = loss / args.accumulate_gradients
                if args.fp16:
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                train_loss_meter.update(loss.item())

                if (batch_step + 1) % args.accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    if args.fp16:
                        # torch.nn.utils.clip_grad_norm_(
                        #     amp.master_params(optimizer), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # torch.nn.utils.clip_grad_norm_(
                        #     model.parameters(), args.max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank == -1 or args.local_rank == 0:
                        tb_logger.add_scalar('batch_train_loss',
                                            loss.item(), global_step)
                        tb_logger.add_scalar('smoothed_train_loss',
                                            train_loss_meter.avg, global_step)
                        tb_logger.add_scalar('ans_span_loss',
                                            output['ans_span_loss'].sum().item(), global_step)
                        if 'sentence_loss' in output:
                            tb_logger.add_scalar('sentence_loss',
                                            output['sentence_loss'].sum().item(), global_step)
                        if 'ans_type_loss' in output:
                            tb_logger.add_scalar('ans_type_loss',
                                            output['ans_type_loss'].sum().item(), global_step)

                    if global_step % log_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):
                        logger.info("Step %d Train loss %.8f on epoch=%d, best_metric=%.3f" % (
                        global_step, train_loss_meter.avg, epoch, best_f1))

                    if args.eval_period_ratio > 0 and global_step % eval_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):

                        metric = predict(tokenizer, model, eval_dataloader, logger, args)
                        pred_list = metric['pred_list']
                        metric = metric['f1']
                        logger.info("Step %d Train loss %.8f score %.3f on epoch=%d" % (
                        global_step, train_loss_meter.avg, metric, epoch))

                        tb_logger.add_scalar('f1',
                                            metric, global_step)
                        if best_f1 < metric:
                            logger.info("Saving model with best score %.3f -> score %.3f on epoch=%d" %
                                        (best_f1, metric, epoch))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            best_f1 = metric
                            json.dump(pred_list, open(os.path.join(args.output_dir, "pred_best.json"), 'w'))

            if args.local_rank == -1 or args.local_rank == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_last.pt"))
                metric = predict(tokenizer, model, eval_dataloader, logger, args)
                pred_list = metric['pred_list']
                metric = metric['f1']
                json.dump(pred_list, open(os.path.join(args.output_dir, "pred_last.json"), 'w'))
                logger.info("Step %d Train loss %.8f f1_score %.8f on epoch=%d" % (
                    global_step, train_loss_meter.avg, metric, epoch))

                tb_logger.add_scalar('f1',
                                            metric, global_step)
                
                if best_f1 < metric:
                    logger.info("Saving model with best score %.3f -> score %.3f on epoch=%d" %
                                (best_f1, metric, epoch))
                    torch.save(model.state_dict(), os.path.join(
                        args.output_dir, f"checkpoint_best.pt"))
                    best_f1 = metric
                    json.dump(pred_list, open(os.path.join(args.output_dir, "pred_best.json"), 'w'))

        logger.info("Training finished!")

    elif args.do_predict:
        metric = predict(tokenizer, model, eval_dataloader, logger, args)
        json.dump(metric['pred_list'], open(os.path.join(args.output_dir, "pred.json"), 'w'))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def predict(tokenizer, model, eval_dataloader, logger, args):
    model.eval()
    logger.info("begin evaluation")
    pred_list = {}
    total_num = 0
    metrics = {"em": 0,
            "f1": 0,
            "prec": 0,
            "recall": 0
            }
    if args.dataset_type == 'hotpot':
        metrics.update({
            "sp_em": 0,
            "sp_f1": 0,
            "sp_prec": 0,
            "sp_recall": 0,
            "joint_em": 0,
            "joint_f1": 0,
            "joint_prec": 0,
            "joint_recall": 0})
        
    for i, batch in enumerate(tqdm(eval_dataloader)):
        id = batch.pop('id')
        answer = batch.pop('answer')
        if 'sentence_num' in batch:
            sentence_num = batch.pop('sentence_num')
            sent_offsets = 0
        batch = move_to_cuda(batch)
        with torch.no_grad():
            outputs = model(**batch)
        start_logits = outputs['start_qa_logits']
        end_logits = outputs['end_qa_logits']

        if 'sentence_select' in outputs:
            sentence_select = torch.argmax(outputs['sentence_select'], axis=-1)
        total_num += start_logits.shape[0]
        for j in range(start_logits.shape[0]):
            all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][j].tolist())
            pred_answer = None
            if 'sentence_select' in outputs:
                output_answer_type = outputs['output_answer_type'][j]
                sentence = sentence_num[j]
                ans_type = torch.argmax(output_answer_type).item()
                if ans_type == 0:
                    pred_answer = 'no'
                elif ans_type == 1:
                    pred_answer = 'yes'
                
                tp, fp, fn = 0, 0, 0
                for s in range(sentence):
                    if sentence_select[s + sent_offsets] == 1:
                        if batch['sentence_labels'][s + sent_offsets] == 1:
                            tp += 1
                        else:
                            fp += 1
                    elif batch['sentence_labels'][s + sent_offsets] == 1:
                        fn += 1
                sent_offsets += sentence
                sp_prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
                sp_recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
                sp_f1 = 2 * sp_prec * sp_recall / (sp_prec + sp_recall) if sp_prec + sp_recall > 0 else 0.0
                sp_em = 1.0 if fp + fn < 1 else 0.0
                metrics["sp_em"] += sp_em
                metrics["sp_f1"] += sp_f1
                metrics["sp_prec"] += sp_prec
                metrics["sp_recall"] += sp_recall
                
            if pred_answer == None:
                answer_tokens = all_tokens[torch.argmax(start_logits[j]) : torch.argmax(end_logits[j]) + 1]
                pred_answer = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(answer_tokens)
                )
                pred_answer = normalize_answer(pred_answer)

            ground_truth_answer = normalize_answer(answer[j])

            pred_list[id[j]] = pred_answer
            em = compute_exact(ground_truth_answer, pred_answer)
            f1, precision, recall = compute_f1(ground_truth_answer, pred_answer)
            metrics["em"] += em
            metrics["f1"] += f1
            metrics["prec"] += precision
            metrics["recall"] += recall

            if args.dataset_type == 'hotpot':
                joint_prec = precision * sp_prec
                joint_recall = recall * sp_recall
                if joint_prec + joint_recall > 0:
                    joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                else:
                    joint_f1 = 0.0
                joint_em = em * sp_em
                metrics["joint_em"] += joint_em
                metrics["joint_f1"] += joint_f1
                metrics["joint_prec"] += joint_prec
                metrics["joint_recall"] += joint_recall

    for k, v in metrics.items():
        metrics[k] = v / total_num
    logger.info(f"evaluated {len(eval_dataloader)} examples...")
    logger.info(f"performance: {metrics}")
    model.train()
    return {'f1':metrics['f1'] if 'joint_f1' not in metrics else metrics['joint_f1'],'pred_list': pred_list}


if __name__ == "__main__":
    main()
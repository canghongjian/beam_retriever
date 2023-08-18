import argparse


def common_args():
    parser =argparse.ArgumentParser()

    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    # model
    parser.add_argument("--model_name", default="model/deberta-v3-base", type=str)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--use_flash_attention", action='store_true')
    parser.add_argument("--flash_attention_type", default='None', type=str)
    parser.add_argument("--dataset_type", default='hotpot', type=str)
    parser.add_argument("--mean_passage_len", default=70, type=int)
    parser.add_argument("--tokenizer_path", type=str, default='model/deberta-v3-base')
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum total sequence length which consists of question and context.")
    parser.add_argument('--use_negative_sampling', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--predict_batch_size", default=1,
                        type=int, help="Total batch size for predictions.")

    # file
    parser.add_argument("--train_file", type=str,
                        default="data/datasets/mrc/hotpotqa/hotpot_train_v1.1.json")
    parser.add_argument("--predict_file", type=str,
                        default="data/datasets/mrc/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")

    return parser


def train_args():
    parser = common_args()

    parser.add_argument("--learning_rate", default=5e-6, type=float, help="learning rate")
    parser.add_argument("--warmupsteps", default=0.1, type=int)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--accumulate_gradients", default=1, type=int)
    parser.add_argument("--num_train_epochs", default=12, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--prefix', type=str, default="default_prefix")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--eval_period', type=int, default=-1)
    parser.add_argument('--eval_period_ratio', type=float, default=-1.0)
    parser.add_argument('--log_period_ratio', type=float, default=0.01)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--stop-drop", default=0, type=float)
    parser.add_argument("--use-adam", action="store_true")
    parser.add_argument("--warmup-ratio", default=0, type=float, help="Linear warmup over warmup_steps.")
    return parser.parse_args()


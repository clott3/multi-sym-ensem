from pathlib import Path
import logging
import os
import uuid
import subprocess

import submitit
import numpy as np
import argparse

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

parser = argparse.ArgumentParser(description='RotNet Training')
parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar100'],
                    help='dataset (imagenet, cifar100)')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=4.8, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--save-freq', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--topk-path', type=str, default='./imagenet_resnet50_top10.pkl',
                    help='path to topk predictions from pre-trained classifier')
parser.add_argument('--checkpoint-dir', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=Path,
                    metavar='LOGDIR', help='path to tensorboard log directory')
parser.add_argument('--rotation', default=0.0, type=float,
                    help="coefficient of rotation loss")
parser.add_argument('--scale', default='0.05,0.14', type=str)

# Training / loss specific parameters
parser.add_argument('--temp', default=0.2, type=float,
                    help='Temperature for InfoNCE loss')
parser.add_argument('--mask-mode', type=str, default='',
                    help='Masking mode (masking out only positives, masking out all others than the topk classes',
                    choices=['pos', 'supcon', 'supcon_all', 'topk', 'topk_sum', 'topk_agg_sum', 'weight_anchor_logits', 'weight_class_logits', 'weight_kernel'])
parser.add_argument('--topk', default=5, type=int, metavar='K',
                    help='Top k classes to use')
parser.add_argument('--topk-only-first', action='store_true', default=False,
                    help='Whether to only use the first block of anchors')
parser.add_argument('--memory-bank', action='store_true', default=False,
                    help='Whether to use memory bank')
parser.add_argument('--mem-size', default=100000, type=int,
                    help='Size of memory bank')
parser.add_argument('--opt-momentum', default=0.9, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--optimizer', default='lars', type=str,
                    help='Optimizer', choices=['lars', 'sgd'])

# Transform
parser.add_argument('--weak-aug', action='store_true', default=False,
                    help='Whether to use augmentation reguarlization (strong & weak augmentation)')

# Slurm setting
parser.add_argument('--ngpus-per-node', default=6, type=int, metavar='N',
                    help='number of gpus per node')
parser.add_argument('--nodes', default=5, type=int, metavar='N',
                    help='number of nodes')
parser.add_argument("--timeout", default=360, type=int,
                    help="Duration of the job")
parser.add_argument("--partition", default="el8", type=str,
                    help="Partition where to submit")

parser.add_argument("--exp", default="SimCLR", type=str,
                    help="Name of experiment")

parser.add_argument("--seed", default=None, type=int,
                    help="seed")
parser.add_argument('--rotinv', action='store_true', default=False)
parser.add_argument('--stylize', type=str, default=None, choices=['inv', 'eq'])
parser.add_argument('--jigsaw', type=str, default=None, choices=['inv', 'eq'])
parser.add_argument('--rotate', type=str, default=None, choices=['inv', 'eq'])

parser.add_argument('--num-jigsaw-per-batch', type=int, default=8)
parser.add_argument('--trans_p', default=0.5, type=float,
                    help="probability of applying transformation for inv versions")
parser.add_argument('--downsize', default=96, type=int,
                    help="downsize images for rot prediction to conserve memory")
                    
parser.add_argument('--train_val_split', default=-1, type=int)
parser.add_argument('--use_smaller_split', action='store_true')
parser.add_argument('--val_perc', default=20, type=int)

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main
        self._setup_gpu_args()
        main.main_worker(self.args.gpu, self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.checkpoint_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(args.job_dir, exist_ok=True)
    init_file = args.job_dir / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def main():
    args = parser.parse_args()
    args.scale = [float(x) for x in args.scale.split(',')]

    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp
    args.job_dir = args.checkpoint_dir

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    get_init_file(args)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus_per_node
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition

    kwargs = {'slurm_gres': f'gpu:{num_gpus_per_node}',}

    executor.update_parameters(
        mem_gb=30 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=24,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 6
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.exp)

    args.dist_url = get_init_file(args).as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    _logger.info("Submitted job_id:", job.job_id)


if __name__ == '__main__':
    main()

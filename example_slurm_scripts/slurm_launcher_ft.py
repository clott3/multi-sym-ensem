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

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')

# Save settings
parser.add_argument('--checkpoint-dir', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=Path,
                    metavar='LOGDIR', help='path to tensorboard log directory')
parser.add_argument('--stats-dir', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
# Slurm setting
parser.add_argument('--ngpus-per-node', default=6, type=int, metavar='N',
                    help='number of gpus per node')
parser.add_argument('--nodes', default=5, type=int, metavar='N',
                    help='number of nodes')
parser.add_argument("--timeout", default=360, type=int,
                    help="Duration of the job")
parser.add_argument("--partition", default="el8", type=str,
                    help="Partition where to submit")

parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=('imagenet', 'imagenet-r',
                    'imagenet-v2', 'inat-1k', 'flowers-102',
                    'food', 'cifar10', 'cifar100'),
                    help='dataset name')
parser.add_argument('--pretrained', nargs="+", default=[], type=str,
                    help='paths to pretrained models')
parser.add_argument('--eval-mode', default='freeze', type=str,
                    choices=('finetune', 'linear_probe', 'freeze'),
                    help='finetune, linear probe, or freeze resnet weights')
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--val-batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size for validation (uses only 1 gpu)')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=1.0, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--lr-scheduler', default='cosine', type=str, metavar='LR-SCH',
                    choices=('cosine'),
                    help='scheduler for learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')

# single gpu training params
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# distributed training params
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# ensemble params
parser.add_argument('--num-ensem', default=1, type=int,
                    help='number of members in the ensemble')
parser.add_argument('--arch', default='resnet50', type=str,
                    choices=('resnet50'),
                    help='architecture for each member in the ensemble')
parser.add_argument('--convert', action='store_true', default=False,
                    help='Whether to convert from single MultiBackbone \
                    checkpoint to use with EnsembleSSL')
parser.add_argument('--combine_sep_ckpts', action='store_true', default=False,
                    help='Whether to combine sep checkpoints from EnsembleSSL')

# misc
parser.add_argument('--seed', default=None, type=int, metavar='S',
                    help='random seed')
parser.add_argument('--exp-id', default='', type=str,
                    help='Experiment ID for saving the outputs/checkpoints into a file')
parser.add_argument('--stats-filename', default='results.txt', type=str,
                    help='Stats filename to aggregate all results')
parser.add_argument('--save_freq', default=20, type=int)


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import eval_ensem
        self._setup_gpu_args()
        eval_ensem.main_worker(self.args.gpu, None, self.args)

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

    args.distributed = True

    args.checkpoint_dir = args.checkpoint_dir / args.exp_id
    args.log_dir = args.log_dir / args.exp_id
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
        mem_gb=40 * num_gpus_per_node,
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

    executor.update_parameters(name=args.exp_id)

    args.dist_url = get_init_file(args).as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    _logger.info("Submitted job_id:", job.job_id)


if __name__ == '__main__':
    main()

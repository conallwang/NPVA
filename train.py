import os
import sys
import argparse
import yaml
import json
import shutil
import datetime
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from dataset.pica_dataset import PiCADataset

from trainer import Trainer
from utils import seed_everything

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--config_path", default="./params.yaml", type=str)
parser.add_argument("--workspace", type=str, required=True)
parser.add_argument("--version", type=str, required=True)
parser.add_argument("--extra_config", type=str, default="{}", required=False)
parser.add_argument("--local_rank", default=0, type=int, help="node rank for distributed training")
args = parser.parse_args()


local_rank = int(args.local_rank)

# Load config yaml file and pre-process params
with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

extra_config = json.loads(args.extra_config)
for k in extra_config.keys():
    assert k in config, k
config.update(extra_config)

# Dump tmp config file
tmp_config_path = os.path.join(os.path.dirname(args.config_path), "params_tmp.yaml")
if local_rank == 0:
    with open(tmp_config_path, "w") as f:
        print("Dumping extra config file...")
        yaml.dump(config, f)

# pre-process params
config["training.gpus"] = [int(s) for s in str(config["training.gpus"]).split(",")]
config["current_epoch"] = 0

# Config gpu
gpus = config["training.gpus"][local_rank]
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
torch.cuda.set_device(gpus)

# dist env
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=1))
world_size = dist.get_world_size()
global_rank = dist.get_rank()
config["global_rank"] = global_rank


def get_dataset(logger):
    # get dataset
    batch_size = config["data.per_gpu_batch_size"]
    train_set = PiCADataset(config)
    if logger:
        logger.info("number of train images: {}".format(len(train_set)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config["data.num_workers"],
        drop_last=True,
    )
    if config["training.errMap"] is not None:
        train_loader.dataset.errMap = np.load(config["training.errMap"])
        train_loader.dataset.random_sample = "err"

    val_set = PiCADataset(config, is_val=True)
    if logger:
        logger.info("number of test images: {}".format(len(val_set)))
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=config["data.num_workers"],
        drop_last=True,
    )

    return train_loader, val_loader


def train():
    seed_everything(42)

    config["local_rank"] = local_rank
    config["world_size"] = world_size

    # Enable cudnn benchmark for speed optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Config logging and tb writer
    logger = None
    if global_rank == 0:
        import logging

        # logging to file and stdout
        config["log_file"] = os.path.join(args.workspace, args.version, "training.log")
        logger = logging.getLogger("NPA")
        file_handler = logging.FileHandler(config["log_file"])
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.handlers = [file_handler, stream_handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info("Training config: {}".format(config))

        # tensorboard summary_writer
        config["tb_writer"] = SummaryWriter(log_dir=config["local_workspace"])

    config["logger"] = logger

    # Init data loader
    train_loader, val_loader = get_dataset(logger)

    trainer = Trainer(config, logger, train_loader)
    trainer.train(train_loader, val_loader, show_time=False)
    # DEBUG
    # trainer.run_eval(val_loader)


def main():
    workspace = os.path.join(args.workspace, args.version)
    config["local_workspace"] = workspace
    if config["global_rank"] == 0:
        # Create sub working dir
        if not os.path.exists(workspace):
            os.makedirs(workspace)
        shutil.copy(tmp_config_path, os.path.join(workspace, "params.yaml"))
    dist.barrier()

    # Start training
    train()


if __name__ == "__main__":
    main()

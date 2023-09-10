import argparse
import torch
import yaml
import os, sys
import cv2
import lpips

import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from dataset.pica_dataset import load_krt, PiCADataset
from utils import seed_everything, visDepthMap, visPositionMap, write_obj
from trainer import Trainer
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser("EVAL METRICS")
parser.add_argument("--img_list", type=str, required=True, help="image path in packed dataset (mini_data)")
parser.add_argument("--checkpoint", type=str, required=True, help="path of model checkpoint")

args = parser.parse_args()

# make sure params.yaml in the same directory with checkpoint
dir_name = os.path.dirname(args.checkpoint)
config_path = os.path.join(dir_name, "params.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["data.test_list"] = args.img_list
chunk = 40000

config["global_rank"] = 0
config["world_size"] = 1
seed_everything(42)


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def mse2psnr(mse):
    return 10 * np.log10(65025 / mse)

def depth_map2normals(depth_map):
    """convert a depth map to normal map

    Args:
        depth_map: HxW
    """
    zy, zx = np.gradient(depth_map)
    normals = np.dstack((-zx, -zy, np.ones_like(depth_map)))

    n = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= n

    normals += 1
    normals /= 2

    return normals


def gammaCorrect(img, dim: int = -1):
    if dim == -1:
        dim = len(img.shape) - 1
    assert img.shape[dim] == 3
    gamma, black, color_scale = 2.0, 3.0 / 255.0, [1.4, 1.1, 1.6]

    if dim == -1:
        dim = len(img.shape) - 1

    scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
    img = img * scale / 1.1
    return np.clip(
        (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0,
        0,
        2,
    )


def eval_metrics(trainer, logger):
    dataset = PiCADataset(config, is_val=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data.num_workers"],
        drop_last=True,
    )
    
    lpips_evaluator = lpips.LPIPS(net="vgg").cuda()

    l_mse = []
    l_lpips = []

    bar = tqdm(range(len(loader)))
    for item in loader:
        sequence_dir, cam, frame_index = item["exp"][0], item["cam_idx"][0], item["frame"][0]

        # image
        photopath = "{}/images".format(config["data.base_dir"])
        path = "{}/{}/{}/{}.png".format(photopath, sequence_dir, cam, frame_index)
        photo_gt = np.asarray(Image.open(path), dtype=np.float32)
        photo_gt = cv2.resize(photo_gt, (config["data.img_w"], config["data.img_h"]), interpolation=cv2.INTER_LINEAR)
        photo_gt = torch.from_numpy(photo_gt[None]) / 255.0
        photo_gt = photo_gt.cuda()

        trainer.set_data(item)
        outputs = trainer.eval_valid_ray(chunk=chunk)

        photo, valid = (
            outputs["render_rgb"],
            outputs["valid_pixel"] * item["mask"].cuda(),
        )

        # compute metrics
        with torch.no_grad():
            photo_pred = photo
            photo_pred[~valid] = 0
            photo_gt[~valid] = 0
            l_mse.append(((photo_pred * 255 - photo_gt * 255) ** 2).mean().item())
            l_lpips.append(lpips_evaluator(photo_pred.permute((0, 3, 1, 2)), photo_gt.permute((0, 3, 1, 2)), normalize=True).item())

        num = len(l_mse)
        if (num > 0) and (num % 2000 == 0):
            avg_mse = np.mean(l_mse)
            avg_psnr = mse2psnr(avg_mse)
            avg_lpips = np.mean(l_lpips)
            
            logger.info("{}/{}".format(num, len(loader)))
            logger.info("   AVG MSE: {}".format(avg_mse))
            logger.info("   AVG PSNR: {}".format(avg_psnr))
            logger.info("   AVG LPIPS: {}".format(avg_lpips))

        bar.update()
        
    avg_mse = np.mean(l_mse)
    avg_psnr = mse2psnr(avg_mse)
    avg_lpips = np.mean(l_lpips)

    logger.info("AVG MSE: {}".format(avg_mse))
    logger.info("AVG PSNR: {}".format(avg_psnr))
    logger.info("AVG LPIPS: {}".format(avg_lpips))
    

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    # If you want to reproduce fully, this item should be set False
    # But that will reduce the performance
    torch.backends.cudnn.benchmark = False

    args.output_dir = os.path.join(dir_name, "evaluation")
    directory(args.output_dir)

    # Config logging and tb writer
    logger = None
    import logging

    # logging to file and stdout
    # config["log_file"] = os.path.join(dir_name, 'test_image.log')
    logger = logging.getLogger("NPA")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    stream_handler.setFormatter(formatter)
    # file_handler = logging.FileHandler(config["log_file"])
    # file_handler.setFormatter(formatter)
    # logger.handlers = [file_handler, stream_handler]
    logger.handlers = [stream_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    config["training.pretrained_checkpoint_path"] = args.checkpoint
    config["logger"] = logger
    config["global_rank"] = 0
    config["local_rank"] = 0

    logger.info("Config: {}".format(config))

    trainer = Trainer(config, logger, is_val=True)
    trainer.set_eval()

    with torch.no_grad():
        eval_metrics(trainer, logger)

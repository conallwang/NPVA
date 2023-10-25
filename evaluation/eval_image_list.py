import argparse
import torch
import yaml
import os, sys
import cv2

import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from dataset.pica_dataset import load_krt, PiCADataset
from utils import seed_everything, visDepthMap, visPositionMap, write_obj
from trainer import Trainer
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser("EVAL IMAGE LIST")
parser.add_argument("--img_list", type=str, required=True, help="image path in packed dataset (mini_data)")
parser.add_argument("--checkpoint", type=str, required=True, help="path of model checkpoint")
parser.add_argument("--full", action="store_true", help="if rendering all pixels")
parser.add_argument("--preview", action="store_true", help="just save part of results")

args = parser.parse_args()

# make sure params.yaml in the same directory with checkpoint
dir_name = os.path.dirname(args.checkpoint)
config_path = os.path.join(dir_name, "params.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["data.test_list"] = args.img_list
chunk = 40000       # depends on GPU memorys

config["global_rank"] = 0
config["world_size"] = 1
seed_everything(42)


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


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


def eval_image(trainer, logger):
    dataset = PiCADataset(config, is_val=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data.num_workers"],
        drop_last=True,
    )

    bar = tqdm(range(len(loader)))
    for item in loader:
        sequence_dir, cam, frame_index = item["exp"][0], item["cam_idx"][0], item["frame"][0]
        print("Process {} / {} / {}".format(sequence_dir, cam, frame_index))

        # image
        photopath = "{}/images".format(config["data.base_dir"])
        path = "{}/{}/{}/{}.png".format(photopath, sequence_dir, cam, frame_index)
        photo_gt = np.asarray(Image.open(path), dtype=np.float32)
        photo_gt = cv2.resize(photo_gt, (config["data.img_w"], config["data.img_h"]), interpolation=cv2.INTER_LINEAR)
        photo_gt = torch.from_numpy(photo_gt[None]) / 255.0
        photo_gt = photo_gt.cuda()

        trainer.set_data(item)
        outputs = trainer.eval_valid_ray(chunk=chunk, full=args.full, debug=True)

        photo, depth, posMap, dispMap, feats, rast_depth, rast_features, valid = (
            outputs["render_rgb"],
            outputs["render_depth"],
            outputs["posMap"],
            outputs["dispMap"],
            outputs["feats"],
            outputs["rast_depth"],
            outputs["rast_features"],
            outputs["valid_pixel"] * item["mask"].cuda(),
        )

        seqdir = os.path.join(args.output_dir, sequence_dir)
        directory(seqdir)
        camdir = os.path.join(seqdir, cam)
        directory(camdir)
        framedir = os.path.join(camdir, frame_index)
        directory(framedir)

        # compute metrics
        savepath = os.path.join(framedir, "metrics.txt")
        with torch.no_grad():
            photo_pred = photo[valid]
            photo_gt = photo_gt[valid].cuda()
            m_mse = ((photo_pred * 255 - photo_gt * 255) ** 2).mean()
        with open(savepath, "w") as f:
            f.write("MSE: {}\n".format(m_mse.item()))

        # save photos
        savepath = os.path.join(framedir, "photo.png")
        photo = photo[0].detach().cpu().numpy()
        cv2.imwrite(savepath, cv2.cvtColor(photo * 255, cv2.COLOR_RGB2BGR))

        # save gamma
        img = cv2.imread(savepath)
        color_corr = lambda x: np.clip(255 * gammaCorrect(x / 255.0, dim=-1), 0, 255)
        img = color_corr(img)
        savedir = os.path.dirname(savepath)
        cv2.imwrite(os.path.join(savedir, "gamma.png"), img)

        # save depth
        savepath = os.path.join(framedir, "depth.png")
        depth = depth[0].detach().cpu().numpy()
        visDepthMap(savepath, depth)

        savepath = os.path.join(framedir, "normals.png")
        normals = depth_map2normals(depth)
        cv2.imwrite(savepath, normals[..., ::-1] * 255)
        
        # save rast_depth
        savepath = os.path.join(framedir, "rast_depth.png")
        rast_depth = rast_depth[0].detach().cpu().numpy()
        visDepthMap(savepath, rast_depth)
        
        savepath = os.path.join(framedir, "rast_normals.png")
        rast_normals = depth_map2normals(rast_depth)
        cv2.imwrite(savepath, rast_normals[..., ::-1] * 255)
        
        # save feature map
        if not args.preview:
            savepath = os.path.join(framedir, "featMap.npy")
            feats = feats[0].detach().cpu().numpy().reshape(1024, 1024, -1)
            np.save(savepath, feats)
            
            savepath = os.path.join(framedir, "rastfeatMap.npy")
            rast_feats = rast_features[0].detach().cpu().numpy()
            np.save(savepath, rast_feats)

        # save posMap
        savepath = os.path.join(framedir, "posMap.png")
        posMap1k = F.interpolate(posMap, scale_factor=4, mode="bilinear", align_corners=False)
        posMap1k = posMap1k[0].detach().cpu().numpy().transpose((1, 2, 0))
        visPositionMap(savepath, posMap1k)
        
        if not args.preview:
            savepath = os.path.join(framedir, "posMap.npy")
            posMap = posMap[0].detach().cpu().numpy().transpose((1, 2, 0))
            np.save(savepath, posMap)
            
            savepath = os.path.join(framedir, "densemesh.obj")
            verts = posMap.reshape(-1, 3)
            tris = np.load('./assets/data/tris.npy')
            write_obj(savepath, verts, tris)
            
            savepath = os.path.join(framedir, "posMap1k.npy")
            np.save(savepath, posMap1k)

            # save points
            savepath = os.path.join(framedir, "geometry.obj")
            vertices = posMap1k.reshape(-1, 3)
            # tris = np.load("./assets/data/tris.npy")
            write_obj(savepath, vertices, log=False)

        if dispMap is not None and not args.preview:
            savepath = os.path.join(framedir, "dispMap.npy")
            dispMap = dispMap[0].detach().cpu().numpy().transpose((1, 2, 0))
            np.save(savepath, dispMap)
            
            savepath = os.path.join(framedir, "geometry_dist.obj")
            posMap1k = posMap1k + dispMap
            vertices = posMap1k.reshape(-1, 3)
            write_obj(savepath, vertices, log=False)

        # save valid pixel
        savepath = os.path.join(framedir, "rast_mask.png")
        rast_mask = valid[0].detach().cpu().numpy()
        cv2.imwrite(savepath, rast_mask * 255)

        bar.update()


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

    # logging to stdout. If you want to log to file, uncomment the following codes.
    # config["log_file"] = os.path.join(dir_name, 'test_image.log')
    logger = logging.getLogger("NPVA")
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
        eval_image(trainer, logger)

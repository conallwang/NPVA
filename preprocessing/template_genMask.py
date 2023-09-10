import numpy as np
import os
import torch
import nvdiffrast.torch as dr

from tqdm import tqdm

root = "/path/to/multiface/m--20190529--1300--002421669--GHS/"
result_folder = "/path/to/multiface/m--20190529--1300--002421669--GHS/gendir"

# camera ids
# For multiface v1 (38 views)
cam_numbers = [400002, 400004, 400007, 400008, 400009, 400010, 400012, 400013, 400015, 400016, \
            400017, 400018, 400019, 400023, 400026, 400027, 400028, 400029, 400030, 400031,  \
            400037, 400039, 400041, 400042, 400048, 400049, 400050, 400051, 400053, 400055,  \
            400059, 400060, 400061, 400063, 400064, 400067, 400069, 400070]

# For multiface v2 (150 views)
# cam_numbers = ['400262', '400263', '400264', '400265', '400266', '400267', '400268', '400269', '400270', '400271', 
#                '400272', '400273', '400274', '400275', '400276', '400279', '400280', '400281', '400282', '400283', 
#                '400284', '400285', '400287', '400288', '400289', '400290', '400291', '400292', '400293', '400294', 
#                '400296', '400297', '400298', '400299', '400300', '400301', '400310', '400312', '400313', '400314', 
#                '400315', '400316', '400317', '400319', '400320', '400321', '400322', '400323', '400324', '400326', 
#                '400327', '400330', '400336', '400337', '400338', '400341', '400342', '400345', '400346', '400347', 
#                '400348', '400349', '400350', '400352', '400353', '400354', '400356', '400357', '400358', '400360', 
#                '400361', '400362', '400363', '400364', '400365', '400366', '400367', '400368', '400369', '400371', 
#                '400372', '400374', '400375', '400377', '400378', '400379', '400380', '400399', '400400', '400401', 
#                '400403', '400404', '400405', '400406', '400407', '400408', '400410', '400411', '400412', '400413', 
#                '400415', '400416', '400417', '400418', '400420', '400421', '400422', '400424', '400425', '400428', 
#                '400430', '400431', '400432', '400433', '400434', '400436', '400437', '400439', '400440', '400441', 
#                '400442', '400447', '400448', '400449', '400450', '400451', '400452', '400453', '400454', '400456', 
#                '400460', '400464', '400469', '400475', '400477', '400479', '400480', '400481', '400482', '400483', 
#                '400484', '400485', '400486', '400487', '400488', '400489', '400500', '400501', '400502', '400503']

torch.cuda.set_device(0)


def load_obj(filename):
    vertices = []
    faces_vertex, faces_uv = [], []
    uvs = []
    with open(filename, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])
    # make sure triangle ids are 0 indexed
    obj = {
        "verts": np.array(vertices, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }
    return obj


def load_krt(path, rate_h=1.0, rate_w=1.0):
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = np.array([[float(x) for x in f.readline().split()] for i in range(3)])
            dist = np.array([float(x) for x in f.readline().split()])
            extrin = np.array([[float(x) for x in f.readline().split()] for i in range(3)])
            f.readline()

            camrotc2w = extrin[:3, :3].T
            campos = -camrotc2w @ extrin[:3, 3]

            intrin[0] *= rate_w
            intrin[1] *= rate_h

            cameras[name[:-1]] = {
                "intrin": intrin,
                "dist": dist,
                "extrin": extrin,
                "camrotc2w": camrotc2w,
                "campos": campos,
            }

    return cameras


def upTo8Multiples(num):
    """

    Args:
        num (int): any number
    """
    res = num
    if num % 8:
        res = 8 * (num // 8 + 1)
    return res


class Renderer:
    def __init__(self, img_h, img_w):
        # self.glctx = dr.RasterizeGLContext()
        self.glctx = dr.RasterizeCudaContext()

        self.resolution = [img_h, img_w]

    def render(self, M, pos, pos_idx):
        ones = torch.ones((pos.shape[0], pos.shape[1], 1)).to(pos.device)
        pos_homo = torch.cat((pos, ones), -1)
        projected = torch.bmm(M, pos_homo.permute(0, 2, 1))
        projected = projected.permute(0, 2, 1)  # [B, N_verts, 3]
        proj = torch.zeros_like(projected)
        proj[..., 0] = (projected[..., 0] / (self.resolution[1] / 2) - projected[..., 2]) / projected[..., 2]
        proj[..., 1] = (projected[..., 1] / (self.resolution[0] / 2) - projected[..., 2]) / projected[..., 2]
        clip_space, _ = torch.max(projected[..., 2], 1, keepdim=True)
        proj[..., 2] = projected[..., 2] / clip_space

        pos_view = torch.cat((proj, torch.ones(proj.shape[0], proj.shape[1], 1).to(proj.device)), -1)
        pos_idx_flat = pos_idx.view(
            (-1, 3)
        ).contiguous()

        rast_out, _ = dr.rasterize(self.glctx, pos_view, pos_idx_flat, self.resolution)

        z = projected[..., 2:]
        rast_attr, _ = dr.interpolate(z, rast_out, pos_idx_flat)

        return rast_out, rast_attr


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def genMask(process_path, img_h=2048, img_w=1334):
    path_split = process_path.split(" ")
    sequence_dir = path_split[0]
    frame_index = path_split[1]

    krt = load_krt(os.path.join(root, "KRT"))
    render = Renderer(upTo8Multiples(img_h), upTo8Multiples(img_w))

    meshpath = os.path.join(root, "tracked_mesh")
    maskpath = os.path.join(result_folder, "rast_mask")
    directory(maskpath)

    exprpath = os.path.join(meshpath, sequence_dir)
    mask_seq_path = os.path.join(maskpath, sequence_dir)
    directory(mask_seq_path)

    obj = load_obj(os.path.join(exprpath, frame_index + ".obj"))
    for cam in cam_numbers:
        cam = str(cam)

        extrin, intrin = krt[cam]["extrin"], krt[cam]["intrin"]
        M = intrin @ extrin

        T_M = torch.from_numpy(M[None].astype(np.float32)).cuda()  # Bx3x4
        T_verts = torch.from_numpy(obj["verts"][None].astype(np.float32)).cuda()  # BxN_vx3
        T_vert_ids = torch.from_numpy(obj["vert_ids"][None].astype(np.int32)).cuda()  # BxN_facex3

        rast_out, _ = render.render(T_M, T_verts, T_vert_ids)
        rast_out = rast_out[:, :img_h, :img_w]
        rast_mask = (rast_out[0, :, :, 3] > 0).detach().cpu().numpy()  # [img_h, img_w]

        if rast_mask.sum() == 0:
            print("[INFO] Wrong Data in {}/{}".format(sequence_dir, frame_index))
            return False
        # coords = np.meshgrid(range(img_w), range(img_h))
        # coords = np.stack(coords, axis=-1)  # [img_h, img_w, 2]

        savedir = os.path.join(mask_seq_path, cam)
        directory(savedir)
        savepath = os.path.join(savedir, frame_index + ".npy")
        np.save(savepath, rast_mask)

    return True


if __name__ == "__main__":
    print("start generating ...")
    with open("path_list.txt", "r") as f:
        lines = f.readlines()

        count = 0
        bar = tqdm(range(len(lines)))
        for line in lines:
            success = genMask(line.strip())
            if not success:
                count += 1
        print("Total wrong data: {}.".format(count))

import numpy as np
import os

from tqdm import tqdm

root = "/path/to/multiface/m--20190529--1300--002421669--GHS/"
result_folder = "/path/to/multiface/m--20190529--1300--002421669--GHS/gendir"


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


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


def ptInTriangle(tri, p):
    """
    Args:
        tri Nx3x2: vertices of triangles
        p 2: query point
    RET:
        N bool, whether the query point is in the triangle (containing edge).
    """
    # f_num = tri.shape[0]
    eps = 1e-9

    AB = tri[:, 1] - tri[:, 0]  # Nx2
    AC = tri[:, 2] - tri[:, 0]
    AP = p - tri[:, 0]

    area = np.cross(AB, AC)  # N
    # sign = np.where(area < 0, -np.ones(f_num), np.ones(f_num))

    s = np.cross(AP, AC) / (area + eps)  # N
    t = np.cross(AB, AP) / (area + eps)

    return s, t, (s >= -1e-4) & (t >= -1e-4) & ((s + t) <= 1 + 1e-4)


def Mesh2PositionMap(obj, pos_size=256, progress=False):
    """~ 45s
    Args:
        obj['verts']:   7306x3, xyz coordinates of vertices
        obj['uvs']:     32808x2, uv coordinates for uv_ids
        obj['vert_ids']:    10936x3, vert_ids to build faces
        obj['uv_ids']:      10936x3, corresponding uv_ids for vert_ids

        pos_size (int, optional): size of postion map. Defaults to 256.
        progress (bool, optional): if showing progress bar. Defaults is False.
    RET:
        posMap:     256x256x3, postion map of mesh
    """
    posMap = np.zeros((pos_size, pos_size, 3))

    tri = obj["uvs"][obj["uv_ids"]] * (pos_size - 1)  # Nx3x2

    if progress:
        bar = tqdm(range(pos_size))
    for u in range(pos_size):
        for v in range(pos_size):
            p = np.array([u, v])
            s, t, searchRes = ptInTriangle(tri, p)

            # print(np.where(searchRes))
            f_id = np.where(searchRes)[0][0]
            bary = np.array([[(1 - s[f_id] - t[f_id])], [s[f_id]], [t[f_id]]])
            face = obj["vert_ids"][f_id]
            posMap[u, v] = (obj["verts"][face] * bary).sum(axis=0)

        if progress:
            bar.update()

    return posMap


def write_obj(filepath, verts, tris):
    """write obj file

    Args:
        verts:      65536x3, vertices coordinates
        tris:       n_facex3, faces consisting of vertices id
    """
    fw = open(filepath, "w")
    # vertices
    for vert in verts:
        fw.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

    for tri in tris:
        fw.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    fw.close()
    print(f"mesh has been saved in {filepath}.")


def generate(process_path):
    path_split = process_path.split(" ")
    sequence_dir = path_split[0]
    frame_index = path_split[1]

    # Create directories
    posMap_dir = os.path.join(result_folder, "posMap")
    directory(posMap_dir)
    posMap_seq_dir = os.path.join(posMap_dir, sequence_dir)
    directory(posMap_seq_dir)

    posMap_check = os.path.join(result_folder, "posmap_check")
    directory(posMap_check)
    posMap_check_seq = os.path.join(posMap_check, sequence_dir)
    directory(posMap_check_seq)

    checkpath = os.path.join(posMap_check_seq, frame_index + ".check")
    if os.path.exists(checkpath):
        return

    # convert coordinates in 'World Frame' to 'Face-centric Frame'
    obj_file = os.path.join(root, "tracked_mesh", sequence_dir, frame_index + ".obj")
    trans_file = os.path.join(root, "tracked_mesh", sequence_dir, frame_index + "_transform.txt")
    obj = load_obj(obj_file)

    trans = np.genfromtxt(trans_file)
    trans_full = np.concatenate((trans, np.array([[0, 0, 0, 1]])))
    trans_inv = np.linalg.inv(trans_full)[:3]  # 3x4

    verts = np.concatenate((obj["verts"], np.ones((obj["verts"].shape[0], 1))), axis=-1).transpose(1, 0)  # 4xN
    obj["verts"] = (trans_inv @ verts).transpose(1, 0)  # Nx3

    # DEBUG
    # write_obj(os.path.join(posMap_seq_dir, frame_index + '_trans.obj'), obj["verts"], obj["vert_ids"] + 1)

    # Coarse Mesh --> Position Map
    # Implementation of Section 4.1 in "A Decoupled 3D Facial Shape Model by Adversarial Training, ICCV'19"
    posMap = Mesh2PositionMap(obj)  # 256x256x3

    np.save(os.path.join(posMap_seq_dir, frame_index + ".npy"), posMap)

    os.system("touch " + checkpath)


if __name__ == "__main__":
    print("start generating posMaps ...")
    with open("path_list.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())
            generate(line.strip())

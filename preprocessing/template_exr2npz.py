import numpy as np
import os
import Metashape
import sparse

root = "/path/to/multiface/m--20190529--1300--002421669--GHS/gendir"

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


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def convert(process_path):
    path_split = process_path.split(" ")
    sequence_dir = path_split[0]
    frame_index = path_split[1]

    depths_seq_dir = os.path.join(root, "depths", sequence_dir)
    # normals_seq_dir = os.path.join(root, 'normals', sequence_dir)

    check_dir = os.path.join(root, "convert_check")
    directory(check_dir)
    check_seq_dir = os.path.join(check_dir, sequence_dir)
    directory(check_seq_dir)

    for cam in cam_numbers:
        cam = str(cam)
        check_cam_dir = os.path.join(check_seq_dir, cam)
        directory(check_cam_dir)
        checkpath = os.path.join(check_cam_dir, frame_index + ".check")
        if os.path.exists(checkpath):
            continue

        depthpath = os.path.join(depths_seq_dir, cam, frame_index + ".exr")
        # normalpath = os.path.join(normals_seq_dir, cam, frame_index + '.exr')
        # if (not os.path.exists(depthpath)) or (not os.path.exists(normalpath)):
        #     continue
        if not os.path.exists(depthpath):
            continue

        try:
            depth_exr = Metashape.Image.open(depthpath, datatype="F32")
            depth_np = np.frombuffer(depth_exr.tostring(), dtype=np.float32).reshape(2048, 1334, 1)
            sdepth = sparse.COO.from_numpy(depth_np)

            # normal_exr = Metashape.Image.open(normalpath, datatype='F32')
            # normal_np = np.frombuffer(normal_exr.tostring(), dtype=np.float32).reshape(2048, 1334, 4)   # RGBA
            # snormal = sparse.COO.from_numpy(normal_np)

            depth_savepath = os.path.join(depths_seq_dir, cam, frame_index + ".npz")
            # normal_savepath = os.path.join(normals_seq_dir, cam, frame_index + '.npz')
            sparse.save_npz(depth_savepath, sdepth)
            # sparse.save_npz(normal_savepath, snormal)

            os.remove(depthpath)
            # os.remove(normalpath)

            os.system("touch " + checkpath)
        except Exception:
            print("Skip {} {} {}".format(sequence_dir, cam, frame_index))
            continue


if __name__ == "__main__":
    print("start converting ...")
    with open("path_list.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())
            convert(line.strip())

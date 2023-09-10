# -*- coding: utf-8 -*-

import os
import time
import threading
import Metashape

import pickle
# import numpy as np

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

use_prior_pose = True

cam_pose_filepath = '/path/to/multiface/m--20190529--1300--002421669--GHS/cam_gen.xml'     # camera config file, obtained from Metashape
result_folder = '/path/to/multiface/m--20190529--1300--002421669--GHS/gendir'
img_root_dir = '/path/to/multiface/m--20190529--1300--002421669--GHS/images'


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + ' exists. (multiprocess conflict)')


def reconstruct(process_path):
    print(process_path)
    path_split = process_path.split(' ')
    sequence_dir = path_split[0]
    frame_index = path_split[1]

    depth_check_dir = os.path.join(result_folder, 'depth_check')
    directory(depth_check_dir)
    depth_check_seq_dir = os.path.join(depth_check_dir, sequence_dir)
    directory(depth_check_seq_dir)
    
    checkpath = os.path.join(depth_check_seq_dir, str(frame_index) + '.check')
    if os.path.exists(checkpath):
        return

    # build directories
    depth_dir = os.path.join(result_folder, 'depths')
    # normals_dir = os.path.join(result_folder, 'normals')
    directory(depth_dir)
    # directory(normals_dir)
    
    depth_seq_dir = os.path.join(depth_dir, sequence_dir)
    # normals_seq_dir = os.path.join(normals_dir, sequence_dir)
    directory(depth_seq_dir)
    # directory(normals_seq_dir)
    
    
    
    # create a chunk
    doc=Metashape.Document()
    chunk=doc.addChunk() 

    frame_img_folder = os.path.join(img_root_dir, sequence_dir)
    
    imgs_path = []
    for cam in cam_numbers:
        img_path = os.path.join(frame_img_folder, str(cam), frame_index + '.png')
        if os.path.exists(img_path) is False:
            raise Exception("Invalid frame, donot have imgs!")
        imgs_path.append(img_path)

    # add multi-view images of one frame
    print(imgs_path)
    chunk.addPhotos(imgs_path)

    # Default image labels are the same as filenames.
    # Here, change image labels to be consistent with camera config file (.xml).
    for camera in chunk.cameras:
        camera.label = camera.photo.path.split('/')[-2]
        print(camera.label)
    print("Script finished")

    if use_prior_pose is True:      # using pose in .xml file, required.     * Ensure the same scale as the ground truth
        chunk.importCameras(cam_pose_filepath)
        chunk.matchPhotos(downscale=1,keypoint_limit = 40000, tiepoint_limit = 10000, generic_preselection=True,reference_preselection=False)
        chunk.triangulatePoints()
    else : 
        chunk.matchPhotos(downscale=0,keypoint_limit = 40000, tiepoint_limit = 10000, generic_preselection=True,reference_preselection=False)

    # aligh cameras
    print('start align......')
    chunk.alignCameras(adaptive_fitting=True)
    print('end align......')

    # Build depth maps
    print('start build depth map......')
    chunk.buildDepthMaps(downscale=1,  reuse_depth=True, filter_mode=Metashape.NoFiltering, cameras=chunk.cameras)
    print('end build depth map......')

    # Build mesh
    chunk.buildModel(source_data = Metashape.DepthMapsData, interpolation=Metashape.EnabledInterpolation, \
        face_count=Metashape.HighFaceCount)
    
    # Apply chunk scale
    if chunk.transform.scale:
        scale = chunk.transform.scale
    else:
        scale = 1
    
    # Save depth maps for each view
    for camera in chunk.cameras:
        if camera.transform is None:
            continue
        
        savepath = os.path.join(depth_seq_dir, camera.label)
        directory(savepath)
        
        depth = chunk.model.renderDepth(camera.transform, camera.sensor.calibration, add_alpha=False)
        img = depth * scale
        
        img.save(os.path.join(savepath, frame_index + '.exr'))
        print("Processed depth for " + camera.label + " " + frame_index)
    
    checkpath = os.path.join(depth_check_seq_dir, str(frame_index) + '.check')
    os.system("touch " + checkpath)
    
    # Check resulting .xml file
    # chunk.exportCameras('new_cam.xml')
    
    # Delete chunk to save space
    doc.remove(chunk)
    


if __name__ == "__main__":
    print('reconstruct start')
    start0 = time.time() 
    count = 1
    with open('path_list.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            print(line)
            start = time.time()
            try:
                print(line.strip())
                reconstruct(line.strip())
                # print('here')
            except Exception as e:
                print(line + ' error!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(e.args)
                print(str(e))
                with open(os.path.join(result_folder, 'error_list.txt'),'a') as f:
                    f.write(line )
            
            end = time.time()
            print(count, '/', len(lines), ' cost time:', end-start)
            count+=1
    
    end0 = time.time()
    print('reconstruct end, ',  count ,' tasks, ', 'cost: ',  end0-start0)

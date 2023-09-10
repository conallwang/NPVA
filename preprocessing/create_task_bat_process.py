# -*- coding: utf-8 -*-

from logging import root
import os
import time
import shutil
import json

workspace = '/path/to/ReconByMetashape'

def divide_task(valid_frame_list, N):
    """Divide framelist into small tasks

    Args:
        valid_frame_list: A list of all frames that need to be processed.
        N: Num. of divided tasks
    """

    frame_count = len(valid_frame_list)
            
    each_task_frame = int( frame_count / N )
    one_more_task = frame_count % N

    base_folder = workspace

    with open(os.path.join(base_folder, 'config.json')) as load_json: 
        json_str = json.load(load_json)
    
    
    for i in range(0, N):
        sub_folder_name = os.path.join(workspace, 'config_' + str(i).zfill(5))
        
        # create sub folder
        if os.path.exists(sub_folder_name) is False:
            os.makedirs(sub_folder_name)

        # create config.json
        json_str['model_local_file_path'] = sub_folder_name

        with open(os.path.join(sub_folder_name, 'config.json'),'w') as f:
            json.dump(json_str, f, indent=4)

        with open(os.path.join(sub_folder_name, 'start.sh'),'w') as f:
            firstline = '/path/to/metashape/agisoft_rlm_linux/rlm &'
            f.writelines(firstline + '\n')
            
            secondline = r"/path/to/metashape/metashape-pro/metashape -r ./reconstruct_" \
                + str(i).zfill(5) + r".py -platform offscreen"
            f.writelines(secondline + '\n')

        # write frames into path_list.txt
        start_index = 0
        end_index = 0
        if i<one_more_task:
            start_index = i* (each_task_frame+1)
            end_index = start_index + each_task_frame + 1
        else:
            start_index = one_more_task *(each_task_frame + 1) + (i-one_more_task)*each_task_frame
            end_index = start_index + each_task_frame

        with open(os.path.join(sub_folder_name, 'path_list.txt'),'w') as f: 
            for ii in  range(start_index, end_index):
                path = valid_frame_list[ii]
                f.write(path)
        f.close()
        
        # copy start.sh file to every sub folder
        shutil.copyfile(base_folder + 'agisoft_server.lic', os.path.join(sub_folder_name,'agisoft_server.lic'))
        shutil.copyfile(base_folder + 'agisoft.lic', os.path.join(sub_folder_name,'agisoft.lic'))
        shutil.copyfile(base_folder + 'template_reconstruct.py', os.path.join(sub_folder_name,'reconstruct_' + str(i).zfill(5) + '.py') )

       
       

if __name__ == "__main__":
    with open('/path/to/multiface/m--20190529--1300--002421669--GHS/remain_list.txt', 'r') as f:
        valid_frame_list = f.readlines()
   
    divide_task(valid_frame_list, 90)

   
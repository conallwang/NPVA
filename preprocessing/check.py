# -*- coding: utf-8 -*-

import os
import time

from tqdm import tqdm

result_folder = '/path/to/multiface/m--20190529--1300--002421669--GHS/gendir'

def check(process_path):
    path_split = process_path.split(' ')
    sequence_dir = path_split[0]
    frame_index = path_split[1]

    models_dir = os.path.join(result_folder, 'convert_check')
    models_seq_dir = os.path.join(models_dir, sequence_dir)

    checkpath = os.path.join(models_seq_dir, str(frame_index) + '.check')
    if not os.path.exists(checkpath):
        print('File not exists: ' + checkpath)
        return False
    return True

if __name__ == "__main__":
    print('check start')
    start0 = time.time() 
    count = 0
    
    respath = '/path/to/datasets/multiface/m--20190529--1300--002421669--GHS/remain_list.txt'
    fw = open(respath, 'w')
    with open('/path/to/datasets/multiface/m--20190529--1300--002421669--GHS/frame_list.txt','r') as f:
        lines=f.readlines()
        bar = tqdm(range(len(lines)))
        for line in lines:
            # print(line)
            start = time.time()
            try:
                # print(line.strip())
                if not check(line.strip()):
                    fw.write(line)
                    count += 1
                # print('here')
            except Exception as e:
                print(line + ' error!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(e.args)
                print(str(e))
                with open(os.path.join(result_folder, 'error_list.txt'),'a') as f:
                    f.write(line )
            bar.update()
    fw.close()
    
    end0 = time.time()
    print('check end, ',  count ,' frames unfinished, ', 'cost: ',  end0-start0)
    print(f'remain frames have been put into: {respath}\n \
        You can change the frame path in template_reconstruct.py and rerun the scripts.')

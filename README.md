# Neural Point-based Volumetric Avatar

## [Project Page](https://conallwang.github.io/npva.github.io/) | Video | [PDF](https://dl.acm.org/doi/pdf/10.1145/3610548.3618204)

This is the official code release for "Neural Point-based Volumetric Avatar: Surface-guided Neural Points for Efficient and Photorealistic Volumetric Head Avatar", SIGGRAPH Asia 2023.

![Teaser Image](./assets/teaser_v5.png)

## Requirements

- Python base
    - Python 3.9
    - PyTorch 1.13.1

- Other packages:
```
pip3 install numpy pyyaml opencv-python tensorboard sparse lpips matplotlib ninja
```

- Install [Nvdiffrast](https://nvlabs.github.io/nvdiffrast/#linux) following their instruction.

- Install [Metashape](https://agisoft.freshdesk.com/support/solutions/articles/31000148930-how-to-install-metashape-stand-alone-python-module) for preprocessing data or something else to perform MVS.

## Usage

### Data Preprocessing

Our data is based on [the Multiface Dataset](https://github.com/facebookresearch/multiface/tree/main), and you can download them using their provided [scripts](https://github.com/facebookresearch/multiface/blob/main/download_dataset.py).

After that, several steps will be following:

1. Use MVS software (e.g., Metashape) to generate ***depth maps*** for each image as the ground truth. (Refer to [preprocessing/template_reconstruct.py](./preprocessing/template_reconstruct.py) and [preprocessing/template_exr2npz.py](./preprocessing/template_exr2npz.py))

2. Convert the coarse meshes (provided by the Multiface Dataset) to ***the position maps*** which are the inputs of our encoder. (Refer to [preprocessing/template_genPosMap.py](./preprocessing/template_genPosMap.py))

3. Generate ***mask*** rasterized by the coarse meshes. (Refer to [preprocessing/template_genMask.py](./preprocessing/template_genMask.py))

4. Compute the mean and variance of the position maps using [preprocessing/compute_posMean.py](./preprocessing/compute_posMean.py)

> Due to the large number of facial data, you'd better employ multiple machines to preform preprocessing in parallel.

The final data is organized as:
```
<identity_dir>
|-- frame_list.txt      # all frames
|-- KRT                 # camera calibrations
|-- images
    |-- E001_Neutral_Eyes_Open      # a sequence
    |-- ...
|-- tracked mesh
    |-- E001_Neutral_Eyes_Open
    |-- ...
|-- unwrapped_uv_1024
    |-- E001_Neutral_Eyes_Open
    |-- ...
|-- gendir              # our generated data
    |-- posMean.npy
    |-- posVar.txt
    |-- depths
        |-- E001_Neutral_Eyes_Open
        |-- ...
    |-- posMap
        |-- E001_Neutral_Eyes_Open
        |-- ...
    |-- rast_mask
        |-- E001_Neutral_Eyes_Open
        |-- ...
    |-- ...             # some dirs used to check
|-- ...                 # other provided data (not be used)
```


### Training

By running [start.sh](./start.sh), you can train our network with ***a single GPU (0)***. 

- ***'DEFAULT_PARAMS'*** shows the config path. By modifing the config file, you can train models for different identities and do some ablation studies.
- ***'WORKSPACE'/'VERSION'*** is the output path. Checkpoints and training logs will be saved there.
- If you want to train models with ***multiple GPUs***, or even ***multiple machines***, you just need modifying the parameters for torch.distributed.launch which is illusrated in [Pytorch Distributed](https://pytorch.org/docs/stable/distributed.html).

### Evaluation

We provide our pretrained models [here](https://drive.google.com/drive/folders/19Ot74JMM3vCrpYCqFJj4PefyBBPL5frj?usp=sharing). You can download the model zip file and unzip them to somewhere (/path/to/models_xxxx).

***To evaluate our visual results***, you need

1. Create a 'test.txt' file with contents like
```
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033395
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033398
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033401
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033404
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033407
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033410
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033413
```
> Note that frame numbers may be different for different identities.
2. Execute the command
```
python evaluataion/eval_image_list.py --img_list test.txt --checkpoint /path/to/models_xxxx/checkpoint_lastest.pth [--full] [--preview]
```

***To evaluate our models quantitatively***, you need

1. Execute the command
```
python evaluation/eval_metrics.py --img_list ./split/seqs_rand_split_xxxx/test_list.txt --checkpoint /path/to/models_xxxx/checkpoint_lastest.pth
```

***To generate talking head***, you need

1. Create a 'test.txt' file with contents like
```
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033395
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033398
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033401
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033404
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033407
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033410
SEN_take_charge_of_choosing_her_bridesmaids_gowns 033413
```

2. Execute the command
```
python evaluation/replay_video.py --img_list test.txt --checkpoint /path/to/models_xxxx/checkpoint_latest.pth --prmesh
```
> This command can generate videos similar to our supplementary materials. By changing parameters in [evaluation/replay_video.py](./evaluation/replay_video.py), you can generate different kinds of videos.

## Citation

If you use any data from this dataset or any code released in this repository, please cite the technical report (https://arxiv.org/abs/2307.05000)

```
@article{DBLP:journals/corr/abs-2307-05000,
  author       = {Cong Wang and
                  Di Kang and
                  Yan{-}Pei Cao and
                  Linchao Bao and
                  Ying Shan and
                  Song{-}Hai Zhang},
  title        = {Neural Point-based Volumetric Avatar: Surface-guided Neural Points
                  for Efficient and Photorealistic Volumetric Head Avatar},
  journal      = {CoRR},
  volume       = {abs/2307.05000},
  year         = {2023}
}
```

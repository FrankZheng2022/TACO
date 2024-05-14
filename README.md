# TACO: Temporal Action-driven Contrastive Learning

This thread contains the original PyTorch implementation of **TACO** for **offline visual RL** from

[TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning](https://arxiv.org/pdf/2306.13229.pdf) by

[Ruijie Zheng](https://ruijiezheng.com), [Xiyao Wang](https://si0wang.github.io)\*, [Yanchao Sun](https://ycsun2017.github.io)\*, [Shuang Ma](https://www.shuangma.me)\*, [Jieyu Zhao](https://jyzhao.net)\*, [Huazhe Xu](http://hxu.rocks)\*, [Hal Daumé III](http://users.umiacs.umd.edu/~hal/)\*, [Furong Huang](https://furong-huang.com)\*


<p align="center">
  <br><img src='media/dmc.gif' width="500"/><br>
   <a href="https://arxiv.org/pdf/2306.13229.pdf">[Paper]</a>&emsp;<a href="https://ruijiezheng.com/project/TACO/index.html">[Website]</a>
</p>


## Method

**TACO** is a simple yet powerful temporal contrastive learning approach that facilitates the concurrent acquisition of latent state and action representations for agents. **TACO** simultaneously learns a state and an action representation by optimizing the mutual information between representations of current states paired with action sequences and representations of the corresponding future states.

<p align="center">
  <img src='media/overview.png' width="750"/>
</p>


## Citation

If you use our method or code in your research, please consider citing the paper as follows:

```
@inproceedings{
zheng2023taco,
title={\${\textbackslash}texttt\{{TACO}\}\$: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning},
author={Ruijie Zheng and Xiyao Wang and Yanchao Sun and Shuang Ma and Jieyu Zhao and Huazhe Xu and Hal Daumé III and Furong Huang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=ezCsMOy1w9}
}

```

## Instructions

Assuming that you already have [MuJoCo](http://www.mujoco.org) installed, install dependencies using `conda`:

```
conda env create -f environment.yml
conda activate taco
```

To download the data:
```
wget https://obj.umiacs.umd.edu/taco-offline/${TASK_NAME}_${DATASET_TYPE}.zip
```
Choose ``${DATASET_TYPE}$`` from ``replay, mediumreplay, medium`` (For example, ``wget https://obj.umiacs.umd.edu/taco-offline/hopper_hop_replay.zip``.)


To train a **DrQv2-BC** agent:
```
CUDA_VISIBLE_DEVICES=X python train.py agent=drqv2_bc task=${TASK_NAME} exp_name=${EXP_NAME} seed=${SEED} offline_data_dir=${OFFLINE_DATA_DIR} 
```

``${OFFLINE_DATA_DIR}`` is where the downloaded offline dataset gets stored. Refer to the `cfgs` directory for a full list of options and default hyperparameters.


## Acknowledgement
TACO is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank DrQ-v2 authors for open-sourcing the [DrQv2](https://github.com/facebookresearch/drqv2) codebase. Our implementation builds on top of their repository.




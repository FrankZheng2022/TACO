# TACO: Temporal Action-driven Contrastive Learning

Original PyTorch implementation of **TACO** from

[TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning](https://arxiv.org/pdf/2306.13229.pdf) by

[Ruijie Zheng](https://ruijiezheng.com), [Xiyao Wang](https://si0wang.github.io)\*, [Yanchao Sun](https://ycsun2017.github.io)\*, [Shuang Ma](https://www.shuangma.me)\*, [Jieyu Zhao](https://jyzhao.net)\*, [Huazhe Xu](http://hxu.rocks)\*, [Hal Daumé III](http://users.umiacs.umd.edu/~hal/)\*, [Furong Huang](https://furong-huang.com)\*


<p align="center">
  <br><img src='media/dmc.gif' width="600"/><br>
   <a href="https://arxiv.org/pdf/2306.13229.pdf">[Paper]</a>&emsp;<a href="https://ruijiezheng.com/project/TACO/index.html">[Website]</a>
</p>


## Method

**TACO** is a simple yet powerful temporal contrastive learning approach that facilitates the concurrent acquisition of latent state and action representations for agents. **TACO** simultaneously learns a state and an action representation by optimizing the mutual information between representations of current states paired with action sequences and representations of the corresponding future states.

<p align="center">
  <img src='media/overview.png' width="600"/>
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
conda env create -f environment.yaml
conda activate tdmpc
```

After installing dependencies, you can train an agent by calling

```
python src/train.py task=dog-run
```

Evaluation videos and model weights can be saved with arguments `save_video=True` and `save_model=True`. Refer to the `cfgs` directory for a full list of options and default hyperparameters, and see `tasks.txt` for a list of supported tasks.





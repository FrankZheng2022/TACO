

# TACO: Temporal Action-driven Contrastive Learning

This is a PyTorch implementation of TACO. The implementation is built upon the original implementation of [DrQ-v2](https://github.com/facebookresearch/drqv2). 

## Dependencies
First install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.
Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

After installing MuJoCo, install dependencies by:
```sh
pip install -r requirements.txt
```

## Training the agent

To train the agent on quadruped run:
```sh
CUDA_VISIBLE_DEVICES=X python -W ignore train_taco.py task=quadruped_run batch_size=1024 exp_name=EXP_NAME 
```

To train the baseline DrQ-v2:
```sh
CUDA_VISIBLE_DEVICES=X python -W ignore train_taco.py task=quadruped_run drqv2=true exp_name=EXP_NAME 
```

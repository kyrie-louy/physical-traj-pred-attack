# A First Physical-World Trajectory Prediction Attack via LiDAR-induced Deceptions in Autonomous Driving

This is an implementation of the paper:

*A First Physical-World Trajectory Prediction Attack via LiDAR-induced Deceptions in Autonomous Driving* ([USENIX](https://www.usenix.org/conference/usenixsecurity24/presentation/lou)/[arXiv](https://arxiv.org/abs/2406.11707))

## Setup

### Install

1. Clone this repository.

2. CD to the `PIXOR_nuscs` directory and create a conda environment for this detector.

   ```bash
   # install dependecies
   cd PIXOR_nuscs
   conda create -n pixor_nuscs python=3.7
   pip install -r requirements.txt
   
   # compile
   cd srcs/preprocess_nuscs
   make
   ```

3. CD to the `Trajectron-plus-plus` and create a conda environment for this predictor.

   ```bash
   # install dependecies
   cd Trajectron-plus-plus
   conda create -n trajectron++ python=3.7
   pip install -r requirements.txt
   ```

### Data preparation

- Download the processed example data of real-world scenes from [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/yanglou3-c_my_cityu_edu_hk/EihNY4PR4yVJqI12uHc20tIB7a1Qo1Hyqurn3yCp48D3nA?e=Xgzyc4)/[Dropbox](https://www.dropbox.com/scl/fo/6uxty2rufv2rhbrfvhijc/AMtLaPWozC_7dj8YFSLMgbE?rlkey=06iqmbs3vc31qwc948n67nwxh&st=elpvh7pu&dl=0)

  - please email me (yanglou3-c@my.cityu.edu.hk) if you can't download the data.

- Put the `data` folder under the `ros_convert` folder

  ```
  attack_prediction
  	ros_convert
  		data
      PIXOR_nuscs
      Trajectron-plus-plus
  ```

## Attack

- Modify the attack script  `ros_convert/attack_realworld.sh`

  1. Change the `root_dir` to your own path where you cloned this repository.
  2. Choose the target scene to attack
     - Uncomment the selected one and comment all the rest scenes.
     - Choices
       1. right clean (adversarial vehicle parked on the right side without attack)
       2. left clean

- Run the attack script

  - `bash ros_convert/attack_realworld.sh`

  - No need  to activate any conda environment (done in the bash script).

## Evaluation

- Modify the evaluation script `ros_convert/eval_realworld.sh`

  1. Change the `root_dir` to your own path where you cloned this repository.

  2. Choose the target scene to evaluate (scene data collected with adversarial object placed)
     - Uncomment the selected one and comment all the rest scenes)
     - Choices: 
       - right scenario: clean, moderate velocity (5km/h), high velocity (10km/h)
       - left scenario: clean

- Run the evaluation script

  - `bash ros_convert/eval_realworld.sh`


## Bibtex
If this work is helpful for your research, please consider citing the following BibTex entry:

```
@inproceedings{lou2024first,
  title={A First $\{$Physical-World$\}$ Trajectory Prediction Attack via $\{$LiDAR-induced$\}$ Deceptions in Autonomous Driving},
  author={Lou, Yang and Zhu, Yi and Song, Qun and Tan, Rui and Qiao, Chunming and Lee, Wei-Bin and Wang, Jianping},
  booktitle={33rd USENIX Security Symposium (USENIX Security 24)},
  pages={6291--6308},
  year={2024}
}
```


## Acknowledgements

Many thanks to the authors of ["Can we use arbitrary objects to attack lidar perception in autonomous driving?"](https://dl.acm.org/doi/abs/10.1145/3460120.3485377) for providing the LiDAR attack implementation.
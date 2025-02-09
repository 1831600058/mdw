### pytorch implementation for "Multi-Channel Hybrid Information Fusion for Anomaly Sound Detection"

![](D:\icassp\DWP\framework.png)

### Installation

```shell
$ conda create -n dwp python=3.11
$ conda activate dwp
$ pip install -r requirements.txt
$ python train.py
```

### Dataset

[DCASE2020 Task2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds) Dataset: 
+ [development dataset](https://zenodo.org/record/3678171)
+ [additional training dataset](https://zenodo.org/record/3727685)
+ [Evaluation dataset](https://zenodo.org/record/3841772)

data path can be set in dataloader


### Model  File

The lab checkpoints are in a floder named model.
The model can be tested against save_path in the config file.
```shell
$ python test.py
```
### Other
The weights can be viewed in the config document.


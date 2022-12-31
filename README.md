# NeurIPS2022_UOT_fine_tuning
Code for "[Improved Fine-Tuning by Better Leveraging Pre-Training Data](https://openreview.net/pdf?id=YTXIIc7cAQ)", NeurIPS 2022

## Requirement
PyTorch >= 1.9.0

## Data preparation
We use eight datasets following the official instructions.


## UOT data selection

Use the bash script in this repository. You can change the arguments in the bash file, such as data path, hyperparameters and result file path.

```
bash xxx.sh
```

## Fine-Tuning with UOT data selection




### Citation
If you use our code in your research, please cite with:

```
@inproceedings{
liu2022improved,
title={Improved Fine-Tuning by Better Leveraging Pre-Training Data},
author={Ziquan Liu and Yi Xu and Yuanhong Xu and Qi Qian and Hao Li and Xiangyang Ji and Antoni B. Chan and Rong Jin},
booktitle={Advances in Neural Information Processing Systems},
year={2022}
}
```

### Acknowledgement
We use [POT: Python Optimal Transport](https://pythonot.github.io/_modules/ot/unbalanced.html) package in the unbalanced optimal transport computation.

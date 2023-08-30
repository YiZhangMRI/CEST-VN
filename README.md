# CEST-VN

Pytorch implementation of CEST-VN paper [*Accelerating Chemical Exchange Saturation Transfer Imaging Using a Model-based Deep Neural Network With Synthetic Training Data*](https://arxiv.org/abs/2205.10265)

If you use this code and provided data, please refer to:

```
@misc{xu2023accelerating,
      title={Accelerating Chemical Exchange Saturation Transfer Imaging Using a Model-based Deep Neural Network With Synthetic Training Data}, 
      author={Jianping Xu and Tao Zu and Yi-Cheng Hsu and Xiaoli Wang and Kannie W. Y. Chan and Yi Zhang},
      year={2023},
      eprint={2205.10265},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph}
}
```

## Getting Started

Setup environment
We provide an environment file `CEST_VN/env.yml`. A new conda environment can be created with 
```
conda env create -f env.yml
```
This will create a working environment named "CEST_VN"
## Datasets
### Scripts
To obtain sufficient CEST raw data for network training, we provide a pipeline based on the Bloch-McConnell model that synthesizes multi-coil CEST k-space data from the publicly available fastMRI[2] brain dataset:

- `Data_Simulation/BM_3pool_simu_normal.m` : Simulate z-spectra for normal tissues using Bloch-McConnell equation. 
- `Data_Simulation/BM_3pool_simu_tumor.m` : Simulate z-spectra for tumor tissues using Bloch-McConnell equation. 
- `Data_Simulation/main.m` : Generate simulated CEST k-space data from FastMRI data.
### Data
- `Data_Simulation/Data/Natural_image`: Several examples of natural-scene image, used to generate textures.
- `Data_Simulation/Data/FastMRI` : Examples of pre-processed FastMRI data (from 2 scans, the central 10 slices were used).
- `Data_Simulation/Data/Z_spectra` : Simulated z-spectra will be stored in this directory.

## Training
An example of network training can be started as follows.
```bash
python main.py --mode train --Resure False --gpus 0,1,2,3 --batch_size 4
```

## Testing
We provide trained network parameters at ./models. An example of network testing can be started as follows.
```bash
python main_test.py --gpus 0 --test_data Healthy.mat --mask Mask_54_96_96_acc_4_New.mat --model model_acc=4.pth --save_name Healthy_acc=4.mat
```

## Acknowledgments
[1]. Hammernik K, Klatzer T, Kobler E, Recht MP, Sodickson DK, Pock T, Knoll F. Learning a variational network for reconstruction of accelerated MRI data. Magnetic Resonance in Medicine 2018;79(6):3055-3071.

[2]. Zbontar J, Knoll F, Sriram A, Murrell T, Huang Z, Muckley MJ, Defazio A, Stern R, Johnson P, Bruno M. fastMRI: An open dataset and benchmarks for accelerated MRI. 2018. arXiv preprint arXiv:1811.08839.

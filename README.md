# PCL

The main code is available, you can see the ```basicsr.models.losses.PCLoss```. For the loss functions of different models, you can directly replace their loss functions with our PCL. Please refer to the experimental section of the paper.

## 1. Create Environment

The basic environment configuration can be referred to [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer), [RetinexMamba](https://github.com/YhuoyuH/RetinexMamba), [HWMNet](https://github.com/FanChiMao/HWMNet), [SNR-Net](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance), [WaveMamba](https://github.com/AlexZou14/Wave-Mamba) and [FourLLIE](https://github.com/wangchx67/FourLLIE). ```causal_conv1d.whl and mamba_ssm.whl``` please download the .whl files from [Google Drive](https://drive.google.com/drive/folders/1IKsjCBSRgcdvYkiqQnGWsqGEbefFV3zI). For example, you can prepare the environments as follow:

```bash
# Retinexformer
git clone https://github.com/caiyuanhao1998/Retinexformer.git
conda create -n torch2 python=3.9 -y
conda activate torch2

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm

And then install BasicSR
python setup.py develop --no_cuda_ext
```
```bash
# RetinexMamba
git clone https://github.com/YhuoyuH/RetinexMamba.git
conda create -n RetinexMamba python=3.8
conda activate RetinexMamba

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy
pip install opencv-python joblib natsort tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
pip install fvcore
pip install causal_conv1d.whl mamba_ssm.whl

And then install BasicSR
python setup.py develop --no_cuda_ext
```

## 2. Prepare Dataset
Download the following datasets:

LOL-v1 [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Baidu Disk](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)

SID [Baidu Disk](https://pan.baidu.com/share/init?surl=HRr-5LJO0V0CWqtoctQp9w) (code: `gplv`), [Google Drive](https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1)

SMID [Baidu Disk](https://pan.baidu.com/share/init?surl=Qol_4GsIjGDR8UT9IRZbBQ) (code: `btux`), [Google Drive](https://drive.google.com/drive/folders/1OV4XgVhipsRqjbp8SYr-4Rpk3mPwvdvG)

MIT-Adobe FiveK [Baidu Disk](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) (code:`cyh2`), [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing), [Official](https://data.csail.mit.edu/graphics/fivek/)

**Note:** 

(1) Please process the raw images of the MIT Adobe FiveK dataset following [the sRGB output mode](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38) or directly download and use the sRGB image pairs processed by us in the [Baidu Disk](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) (code:`cyh2`) and [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing)

(2) Please download the `text_list.txt` from [Google Drive](https://drive.google.com/file/d/199qrfizUeZfgq3qVjrM74mZ_nlacgwiP/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1GQfaQLI6tvB0IrTMPOM_9Q?pwd=ggbh) (code: `ggbh`) and then put it into the folder `data/SMID/SMID_Long_np/`

<details close>
<summary><b> Then organize these datasets as follows: </b></summary>

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...

    |    |--SID
    |    |    |--short_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--long_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |--SMID
    |    |    |--SMID_LQ_np
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--SMID_Long_np
    |    |    |    |--text_list.txt
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |--FiveK
    |    |    |--train
    |    |    |    |--input
    |    |    |    |    |--a0099-kme_264.jpg
    |    |    |    |    |--a0101-kme_610.jpg
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--a0099-kme_264.jpg
    |    |    |    |    |--a0101-kme_610.jpg
    |    |    |    |     ...
    |    |    |--test
    |    |    |    |--input
    |    |    |    |    |--a4574-DSC_0038.jpg
    |    |    |    |    |--a4576-DSC_0217.jpg
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--a4574-DSC_0038.jpg
    |    |    |    |    |--a4576-DSC_0217.jpg
    |    |    |    |     ...


```

</details>

We also provide download links for LIME, NPE, MEF, DICM, and VV datasets that have no ground truth:

[Baidu Disk](https://pan.baidu.com/s/1oHg03tOfWWLp4q1R6rlzww?pwd=cyh2) (code: `cyh2`)
 or [Google Drive](https://drive.google.com/drive/folders/1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp?usp=drive_link)

&nbsp;                    


## 3. Testing


Download our models from [Baidu Disk](https://pan.baidu.com/s/1HAqnf97CLCHIsG5m9x30uA?pwd=3kz5) (code: `3kz5`) or [Google Drive](https://drive.google.com/drive/folders/1dA1V9VxtVpbD9x0SjSUzdsgSLLSMLMaN?usp=drive_link). Any deviation between some weights and the paper’s values stems from better results obtained in later optimizations.
We use RetinexFormer as an example. For other models, please refer to ([RetinexMamba](https://github.com/YhuoyuH/RetinexMamba), [HWMNet](https://github.com/FanChiMao/HWMNet), [SNR-Net](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance), [WaveMamba](https://github.com/AlexZou14/Wave-Mamba) and [FourLLIE](https://github.com/wangchx67/FourLLIE).

```shell
# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic

# SID
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SID.yml --weights pretrained_weights/SID.pth --dataset SID

# SMID
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SMID.yml --weights pretrained_weights/SMID.pth --dataset SMID

# FiveK
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_FiveK.yml --weights pretrained_weights/FiveK.pth --dataset FiveK

```
*It should be noted that our loss function has achieved a greater improvement in RetineMamba, and it may still have higher performance potential.



## Acknowledgement
This project is based on [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer), [RetinexMamba](https://github.com/YhuoyuH/RetinexMamba), [HWMNet](https://github.com/FanChiMao/HWMNet), [SNR-Net](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance), [WaveMamba](https://github.com/AlexZou14/Wave-Mamba), [FourLLIE](https://github.com/wangchx67/FourLLIE),[DMFourLLIE](https://github.com/bywlzts/DMFourLLIE) and [DarkIR](https://github.com/cidautai/DarkIR).

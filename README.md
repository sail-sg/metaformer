# [MetaFormer Baselines for Vision](https://arxiv.org/abs/2210.13452)


This is a PyTorch implementation of several MetaFormer baslines including **IdentityFormer**, **RandFormer**, **RandFormer** and **CAFormer** proposed by our paper "[MetaFormer Baselines for Vision](https://arxiv.org/abs/2210.13452)".

![Figure1](https://user-images.githubusercontent.com/49296856/197580831-fc937e24-9941-4794-b99d-822748fa0f11.png)
Figure 1: **Performance of MetaFormer baselines and other state-of-the-art models on ImageNet-1K at 224x224 resolution.** The architectures of our proposed models are shown in Figure 2. (a) IdentityFormer/RandFormer achieve over 80%/81% accuracy, indicating MetaFormer has solid lower bound of performance and works well on arbitrary token mixers. The accuracy of well-trained ResNet-50 is from "ResNet strikes back". (b) Without novel token mixers, pure CNN-based ConvFormer outperforms ConvNeXt, while CAFormer sets a new record of 85.5% accuracy on ImageNet-1K at 224x224 resolution under normal supervised training without external data or distillation.

![Overall](https://user-images.githubusercontent.com/15921929/197650554-f387ceb0-fcd4-4e04-9523-1a33be9e9305.png)
Figure 2: **(a-d)  Overall frameworks of IdentityFormer, RandForemr, ConvFormer and CAFormer.** Similar to ResNet, the models adopt hierarchical architecture of 4 stages, and stage $i$ has  $L_i$ blocks with feature dimension $D_i$. Each downsampling module is implemented by a layer of convolution. The first downsampling has kernel size of 7 and stride of 4, while the last three ones have kernel size of 3 and stride of 2. **(e-h) Architectures of IdentityFormer, RandForemr, ConvFormer and Transformer blocks**, which have token mixer of identity mapping, global random mixing, separable depthwise convolutions, or vanilla self-attention, respectively. 

![Comparision](https://user-images.githubusercontent.com/49296856/197601575-6a19ed8c-7bc2-433b-895b-e5363358ea77.png)



## Bibtex
```
@article{yu2022metaformer,
  title={MetaFormer Is Actually What You Need for Vision},
  author={Yu, Weihao and Si, Chenyang and Zhou, Pan and Luo, Mi and Zhou, Yichen and Feng, Jiashi and Yan, Shuicheng and Wang, Xinchao},
  journal={arXiv preprint arXiv:2210.13452},
  year={2022}
}
```



## Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; [timm](https://github.com/rwightman/pytorch-image-models) (`pip install timm==0.6.11`)

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## MetaFormer baselines 
### Models with common token mixers trained and finetuned (at 384) on ImageNet
| Model | Resolution | Params | MACs | Top1 Acc | Download |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |  :---:  |
| caformer_s18 | 224 | 26M | 4.1G |  83.6 | [here](https://drive.google.com/file/d/1lKImHTCrrTsrf0TxK8Lx0XDnxmHzEVkN/view?usp=sharing) |
| caformer_s18_384 | 384 | 26M | 13.4G |  85.0 | [here](https://drive.google.com/file/d/1KHNWFkUwkRO6ggq465hv4NEqNuviKTGc/view?usp=sharing) |
| caformer_s36 | 224 | 39M | 8.0G |  84.5 | [here](https://drive.google.com/file/d/1wYoSANfa8-QigqizR8fKOBwUNbmInpX_/view?usp=sharing) |
| caformer_s36_384 | 384 | 39M | 26.0G |  85.7 | [here](https://drive.google.com/file/d/1_8BLLla1OgqD_Uk8WuC-g60tz0jcbxul/view?usp=sharing) |
| caformer_m36 | 224 | 56M | 13.2G |  85.2 | [here](https://drive.google.com/file/d/1ETqRVoMn689JWp5mf9RH4z7zKBzi-i6B/view?usp=sharing) |
| caformer_m36_384 | 384 | 56M | 42.0G |  86.2 | [here](https://drive.google.com/file/d/1zNODcDBoawl8ZcH1007N2J4gSeo2tmAb/view?usp=sharing) |
| caformer_b36 | 224 | 99M | 23.2G |  85.5 | [here](https://drive.google.com/file/d/1GFGBpn0uuoPzVv0ymHow8hCXRdyaO8hY/view?usp=sharing) |
| caformer_b36_384 | 384 | 99M | 72.2G |  86.4 | [here](https://drive.google.com/file/d/1yGV2aD2CSuirf7yfiP3fs3KyoBCs7RE7/view?usp=sharing) |
| convformer_s18 | 224 | 27M | 3.9G |  83.0 | [here](https://drive.google.com/file/d/16LkvaU8cFh9Cvdoeoa1qyKGWKSXg2pql/view?usp=sharing) |
| convformer_s18_384 | 384 | 27M | 11.6G |  84.4 | [here](https://drive.google.com/file/d/1joDSEtqI3d2Pb_q4V-Dz2ELlWZZ41YCZ/view?usp=sharing) |
| convformer_s36 | 224 | 40M | 7.6G |  84.1 | [here](https://drive.google.com/file/d/1TEcPHGr5gEjL44v68whSGuANgaohTniQ/view?usp=sharing) |
| convformer_s36_384 | 384 | 40M | 22.4G |  85.4 | [here](https://drive.google.com/file/d/1BrMLbSzY4fCRpCmRbqfkkZd82ltEKXyc/view?usp=sharing) |
| convformer_m36 | 224 | 57M | 12.8G |  84.5 | [here](https://drive.google.com/file/d/1QrVYRC8pU0gcmfN2MSESSNgWtrX4QEhm/view?usp=sharing) |
| convformer_m36_384 | 384 | 57M | 37.7G |  85.6 | [here](https://drive.google.com/file/d/1_xYVwVdptRN2ZJC_vIa4A0lkALPHmapy/view?usp=sharing) |
| convformer_b36 | 224 | 100M | 22.6G |  84.8 | [here](https://drive.google.com/file/d/1p8dgr5eREmujD_5V5KoZAtRVlFhJ6S04/view?usp=sharing) |
| convformer_b36_384 | 384 | 100M | 66.5G |  85.7 | [here](https://drive.google.com/file/d/1pmN9K67RfDK723KdGCIEyFh0FJYG3Efb/view?usp=sharing) |


### Models with common token mixers pretrained on ImageNet-21k and finetuned on ImgeNet-1K
| Model | Resolution | Params | MACs | Top1 Acc | Download |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |  :---:  |
| caformer_b36_in21ft1k | 224 | 99M | 23.2G |  87.4 | [here](https://drive.google.com/file/d/1u0vTyl4ZUE08NJduVWDRRZFCjgHJAm0s/view?usp=sharing) |
| caformer_b36_384_in21ft1k | 384 | 99M | 72.2G |  88.1 | [here](https://drive.google.com/file/d/1IgI_0iflgUgr_IC7HDNTtisS1yHha5lZ/view?usp=sharing) |
| convformer_b36_in21ft1k | 224 | 100M | 22.6G |  87.0 | [here](https://drive.google.com/file/d/1nXR9f0nn3qovIh6DH3XEmJpyF_IYQSiT/view?usp=sharing) |
| convformer_b36_384_in21kft1k | 384 | 100M | 66.5G |  87.6 | [here](https://drive.google.com/file/d/1Tsd_mWrQF9Ihzb7T4wcd6tjnC3ndi7ou/view?usp=sharing) |


### Models with common token mixers pretrained on ImageNet-21k
| Model | Resolution | Params | MACs | Download |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |
| caformer_b36_in21k | 224 | 99M | 23.2G | [here](https://drive.google.com/file/d/1RLO4Fp6N9s-B_G46S2tl_NVLlqvpjvsK/view?usp=sharing) |
| convformer_b36_in21k | 224 | 100M | 22.6G | [here](https://drive.google.com/file/d/1O5vU80VD9V-6DIiLbnffZz1Z8iaAN3Pv/view?usp=sharing) |


### Models with basic token mixers trained on ImageNet-1K
| Model | Resolution | Params | MACs | Top1 Acc | Download |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |  :---:  |
| identityformer_s12 | 224 | 11.9M | 1.8G |  74.6 | [here](https://drive.google.com/file/d/1eihh3U3B9M0sp30PyDiX3XqSvstVhAh9/view?usp=sharing) |
| identityformer_s24 | 224 | 21.3M | 3.4G |  78.2 | [here](https://drive.google.com/file/d/1Q6TG-4pTG-HPRmxsjhGd6XKOWy5dGj0g/view?usp=sharing) |
| identityformer_s36 | 224 | 30.8M | 5.0G |  79.3 | [here](https://drive.google.com/file/d/13T0BruraT4k4Z9l_XzKpIxXMwWKCWnDy/view?usp=sharing) |
| identityformer_m36 | 224 | 56.1M | 8.8G |  80.0 | [here](https://drive.google.com/file/d/1-lBG5i6zQRruStH3QJ3pK7lzIL1QiA0d/view?usp=sharing) |
| identityformer_m48 | 224 | 73.3 | 11.5G |  80.4 | [here](https://drive.google.com/file/d/1tWP9bdX_sAUt7mPhEJO9acjSu_tz7dpo/view?usp=sharing) |
| randformer_s12 | 224 | 11.9 + <u>0.2</u>M | 1.9G |  74.6 | [here](https://drive.google.com/file/d/14_xyFSn3jp-K2B1x4gBY7XEmeCntcV24/view?usp=sharing) |
| randformer_s24 | 224 | 21.3 + <u>0.5</u>M | 3.5G |  78.2 | [here](https://drive.google.com/file/d/1rzkTiV7Q8_YhHjnUk7upVHI-O8kJTMo2/view?usp=sharing) |
| randformer_s36 | 224 | 30.8 + <u>0.7</u>M | 5.2G |  79.3 | [here](https://drive.google.com/file/d/16p5hUvO626Uo1amNyG0PUsKi7nPgvLq3/view?usp=sharing) |
| randformer_m36 | 224 | 56.1 + <u>0.7</u>M | 9.0G |  80.0 | [here](https://drive.google.com/file/d/1-1jsgCxtgZumPkwf5qj7Ugg7KpXM8qtb/view?usp=sharing) |
| randformer_m48 | 224 | 73.3 + <u>0.9</u>M | 11.9G |  80.4 | [here](https://drive.google.com/file/d/1wGpT6ctqTPkIFBYxiy0NX5ZqT8hSKM5g/view?usp=sharing) |
| poolformerv2_s12 | 224 | 11.9M | 1.8G |  78.0 | [here](https://drive.google.com/file/d/1BjKY5JWlk2cagSkiSDtpBEmV4dJZj5F1/view?usp=sharing) |
| poolformerv2_s24 | 224 | 21.3M | 3.4G |  80.7 | [here](https://drive.google.com/file/d/1pZzKAoloY0-ph7eflFGvJrCB_c09IXta/view?usp=sharing) |
| poolformerv2_s36 | 224 | 30.8M | 5.0G |  81.6 | [here](https://drive.google.com/file/d/1Z09FBpbF_reAKjtjf_w_e5JGHN8968S_/view?usp=sharing) |
| poolformerv2_m36 | 224 | 56.1M | 8.8G |  82.2 | [here](https://drive.google.com/file/d/1LDG0b19p4p8CywfRtI-qDVDxd1SUf6s5/view?usp=sharing) |
| poolformerv2_m48 | 224 | 73.3 | 11.5G |  82.6 | [here](https://drive.google.com/file/d/1afJe393F_HG5BcGDwkW3bleqvHpYLNHT/view?usp=sharing) |

The underlined numbers mean the numbers of parameters that are frozen after random initialization.

## Validation

To evaluate our PoolFormer models, run:

```bash
MODEL=caformer_s18
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --checkpoint /path/to/checkpoint 
```



## Train
We use batch size of 4096 by default and we show how to train models with 8 GPUs. For multi-node training, adjust `--grad-accum-steps` according to your situations.


```bash
DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/metaformer # modeify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model convformer_s18 --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.2 --head-dropout 0.0
```
Training (fine-tuning) scripts of other models are shown in [scripts](/scripts/).

## Acknowledgment
Weihao Yu would like to thank TPU Research Cloud (TRC) program for the support of partial computational resources.
Our implementation is mainly based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models). 
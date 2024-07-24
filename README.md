# ControlNet-Acceleration-on-RadeonProW7900
Accelerating ControlNet Pipeline through ROCm, Pytorch etc. Making it comparable performance as A100.


# Contents
- [Task Brief](#task-brief)
  - [Controlnet Introductions](#controlnet-introductions)
  - [Stable Diffusion + ControlNet](#stable-diffusion--controlnet)
  - [How to Accelerate this generation pipeline](#how-to-accelerate-this-generation-pipeline)
- [Environment Settings](#environment-settings)
  - [Hardware Preparation and OS Installation](#hardware-preparation-and-os-installation)
  - [Software Preparations](#software-preparations) 

## Task Brief
### ControlNet Introductions
intro based on Original [Code Repo](https://github.com/lllyasviel/ControlNet).
ControlNet 1.0 [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) is a neural network structure to control diffusion models by adding extra conditions.

![img](github_page/he.png)

It copys the weights of neural network blocks into a "locked" copy and a "trainable" copy. 

The "trainable" one learns your condition. The "locked" one preserves your model. 

Thanks to this, training with small dataset of image pairs will not destroy the production-ready diffusion models.

The "zero convolution" is 1×1 convolution with both weight and bias initialized as zeros. 

Before training, all zero convolutions output zeros, and ControlNet will not cause any distortion.

No layer is trained from scratch. You are still fine-tuning. Your original model is safe. 

This allows training on small-scale or even personal devices.

This is also friendly to merge/replacement/offsetting of models/weights/blocks/layers.

### Stable Diffusion + ControlNet
By repeating the above simple structure 14 times, we can control stable diffusion in this way:

![img](github_page/sd.png)

In this way, the ControlNet can **reuse** the SD encoder as a **deep, strong, robust, and powerful backbone** to learn diverse controls. Many evidences (like [this](https://jerryxu.net/ODISE/) and [this](https://vpd.ivg-research.xyz/)) validate that the SD encoder is an excellent backbone.

### How to Accelerate this generation pipeline

## Environment Settings
### Hardware Preparation and OS Installation
#### Hardware elements 
As for my choice of hardware, I consult the product brief on AMD offical websites about RadeonPro W7900(about 295W in docs, but I got 241W in rocm-smi), and pay more attention on stability of Power Supply, which I used a **1000W** in my case.

I rent a server with AMD 3945WX and 128GB RAM, and put the RadeonPro W7900 in the PCIe slot.

Here is system info:
```shell
❯ lscpu
...
CPU(s):                  24
  On-line CPU(s) list:   0-23
Vendor ID:               AuthenticAMD
  Model name:            AMD Ryzen Threadripper PRO 3945WX 12-Cores
...
❯ free -h
               total        used        free      shared  buff/cache   available
Mem:           125Gi        35Gi       951Mi       9.0Mi        89Gi        89Gi
Swap:           14Gi        14Gi          0B
```

#### Power Settings
And put it in a data center with cooling systems which can control the environment temperature to about **19** degree celcius, to reducing the heat pressure, I used [LACT](https://github.com/ilya-zlobintsev/LACT) to surpress the powerCap to **220W**, and finally got about 73 degree celcius for this card.
![img](github_page/PwrCap.png)
Note that this is only the apparent Pwr Consumption of the card, and the actual peak power consumption is more than this.
#### OS version and driver choices
The os versioni that shown in uname -a is:
```shell
❯ uname -a
6.2.0-26-generic #26~22.04.1-Ubuntu
```
The version of RadeonPro for Enterprise GPU Driver is, and can be found [here](https://www.amd.com/en/support/download/linux-drivers.html):
```shell 
amdgpu-install_6.0.60002-1_all.deb # my version# 

# install commands from AMD offical website with ROCm6.1.3
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo apt install ./amdgpu-install_6.1.60103-1_all.deb
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video $LOGNAME
```
#### Set Groups permissions
```shell
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```
#### Post-install verification checks
Verify that the current user is added to the render and video groups.
```shell
# 1 check groups
groups
#output
<username> adm cdrom sudo dip video plugdev render lpadmin lxd sambashare
```

Check if amdgpu kernel driver is installed.
```shell
# 2 check gpu driver kernel module
dkms status

#output
amdgpu/x.x.x-xxxxxxx.xx.xx, x.x.x-xx-generic, x86_64: installed
```
Finally, check rocminfo:
```shell
# 3 check rocminfo (very important and can affect the following steps, especially the pytorch cuda support)
❯ rocminfo

# output
[...]
*******
Agent 2
*******
  Name:                    gfx1100
  Uuid:                    GPU-5ecee39292e80c37
  Marketing Name:          Radeon RX 7900 XTX
  Vendor Name:             AMD
  [...]
[...]
```

### Software Preparations
#### ROCm installation

#### Clone this repo
```shell
git clone https://github.com/jedibobo/ControlNet-Acceleration-on-RadeonProW7900.git
```
#### Conda Environment
First create a new conda environment

    conda env create -f environment.yaml
    conda activate control

Second replace the nvdia cuda pytorch to ROCm enabled Pytorch. The command can be found in Pytorch offical website.
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```
Third verify pytorch installation and GPU support:
```shell
❯ python
Python 3.9.19 (main, Mar 21 2024, 17:11:28) 
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'AMD Radeon PRO W7900'
```

FAQ:

1.If return "False" in torch.cuda.is_available(), please first check if "rocminfo" command returns valid information instead of "no permission" errors. If the latter comes up, please refer to this [link](https://github.com/ROCm/ROCm/issues/1211) and this [link](https://github.com/ROCm/ROCm/issues/1798) to resolve this issue.

#### Model Downloading
Download model control_sd_canny.pth  from [huggingface](https://huggingface.co/lllyasviel/ControlNet/tree/main/models), and place it in models dir.



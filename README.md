<!--- 

export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

-->
Sum-Product Networks for Robust Automatic Speaker Recognition
====
Please note that this repository is currently in its early stages
-----

Sum-product networks are used here for robust speaker recognition. The marginalisation missing feature approach is used as the model-compensation technique to improve the robustness. 

SPFlow is modified to accomadate the implementation.


|![](./spk_model.jpg "SPN speaker model.")|
|----|
| <p align="center"> <b>Figure 1:</b> <a> SPN speaker model.</a> </p> |

TIMIT corpus
-----
The TIMTI corpus is used for the  

Installation
-----

Optional, only required if using a GPU:

* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn)

To install:

1. `git clone https://github.com/anicolson/SPN-Spk-Rec.git`
2. `cd SPN-Spk-Rec`
3. `virtualenv --system-site-packages -p python3 ~/venv/SPN-Spk-Rec`
4. `source ~/venv/SPN-Spk-Rec/bin/activate`
5. `pip install --upgrade tensorflow`
6. `pip install -r requirements.txt`

If a GPU is to be used, replace step 4 with: `pip install --upgrade tensorflow-gpu`

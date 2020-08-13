<!---

export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

-->
[SPNs for robust ASI](https://arxiv.org/abs/1910.11969)
====

Sum-product networks (SPNs) with Gaussian leaves are used as speaker models for automatic speaker identification (ASI) [[1]](https://arxiv.org/abs/1910.11969). An example of an SPN with univariate Gaussian leaves is shown in Figure 1. Marginalisation and bounded marginalisation, as proposed by [Cook *et al.*](https://doi.org/10.1016/S0167-6393(00)00034-0), are used to significantly increase the robustness of the SPN speaker models to noise. To identify the reliable spectral components for marginalisation, an *a priori* SNR estimator is used.

|![](./spk_model.jpg "SPN speaker model.")|
|----|
| <p align="center"> <b>Figure 1:</b> <a> SPN speaker model with univariate Gaussian leaves.</a> </p> |

Installation
-----
1. `git clone https://github.com/anicolson/SPN-ASI.git`
2. `cd SPN-ASI`
3. `virtualenv --system-site-packages -p python3 ~/venv/SPN-ASI`
4. `source ~/venv/SPN-Spk-Rec/bin/activate`
6. `pip install -r requirements.txt`

Dataset
-----
**TIMIT corpus for clean speech:**

The clean speech of the speakers included in the TIMIT corpus are used to train the ASI system. The path to the TIMIT corpus is set in [`config.sh`]().

**Noisy speech and *a priori* SNR estimates:**

The noisy speech and *a priori* SNR estimates used for testing the ASI system. **The noisy speech and *a priori* SNR estimates can be obtained from: [http://dx.doi.org/10.21227/nbec-q510](http://dx.doi.org/10.21227/nbec-q510).** Please see the [Deep Xi](https://github.com/anicolson/DeepXi) repository if you require outputs from an *a priori* SNR estimator. The paths to the noisy speech and *a priori* SNR estimates are set in [`config.sh`]().

How to use the ASI system
-----
**Training:**

```
./run.sh TRAIN 1
```


**Identification:**

```
./run.sh IDENTIFICATION 1 MARG 1 BOUNDS 1
```
`MARG` is the flag for the marginalisation, and `BOUNDS` specifies whether bounds are to be used with marginalisation.


References
-----
Please cite the following:

[[1] Nicolson, A., & Paliwal, K. K. (2020). Sum-Product Networks for Robust Automatic Speaker Identification. Proc. Interspeech 2020](https://arxiv.org/abs/1910.11969).

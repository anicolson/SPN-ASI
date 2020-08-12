<!--- 

export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

-->
SPNs for Robust ASI
====

Sum-product networks (SPNs) with Gaussian leaves are used here as speaker models for automatic speaker identification (ASI). An example of an SPN with univariate Gaussian leaves is shown in Figure 1. Marginalisation and bounded marginalisation, as proposed by Cook *et al.* ([1](https://doi.org/10.1016/S0167-6393(00)00034-0)), is used to significantly increase the robustness of the SPN speaker models to noise. To identify the reliable spectral components for marginalisation, an *a priori* SNR estimator is used.

|![](./spk_model.jpg "SPN speaker model.")|
|----|
| <p align="center"> <b>Figure 1:</b> <a> SPN speaker model with univariate Gaussian leaves.</a> </p> |

Installation
-----

To install:

1. `git clone https://github.com/anicolson/SPN-ASI.git`
2. `cd SPN-ASI`
3. `virtualenv --system-site-packages -p python3 ~/venv/SPN-ASI`
4. `source ~/venv/SPN-Spk-Rec/bin/activate`
6. `pip install -r requirements.txt`

If a GPU is to be used, replace step 4 with: `pip install --upgrade tensorflow-gpu`

Dataset
-----
Place the TIMIT corpus in the data path. E.g. the train directory for TIMIT should be located at data/timit/train. Similarly, the test directory for TIMIT should be located at data/timit/test.

The noisy speech created from the clean speech files from the TIMIT corpus, and the noise of your choosing should be placed in data/noisy_speech. 

Each filename in data/noisy_speech should coprise of the following: *w_x_y_zdB.wav*, where **w** is the speaker (e.g. *mmds0*), **x** is the utterance (e.g. *sa1*), **y** is the noise source (e.g. *f16*), and **z** is the SNR level in dB (e.g. *-5*). An example filename is as follows: *mjjj0_sa1_voice-babble_-5dB.wav*. 

How to use
-----
**Inference on clean speech:**

Note: the TIMIT corpus must be in the correct directory prior to running the following:

```
python3 main.py --ver '0a' --test_clean_speech 1 
```
Version *0a* is what is used in the paper.

**Inference on noisy speech:**

Note: the TIMIT corpus and the noisy speech must be in the correct directories prior to running the following:

```
python3 main.py --ver '0a' --test_noisy_speech 1 --mft 'bmarg'
```
Use *--mft 'bmarg'*, *--mft 'marg'*, or *--mft 'none'* to detirmine what type of marginalisation is used.


**Training:**

Note: the TIMIT corpus must be in the correct directory prior to running the following:

```
python3 main.py --ver 'new_models' --train 1 --verbose 1 --min_instances_slice 50 --threshold 0.3
```

<!--- 


References
-----
Please cite the following when using Deep Xi:

[1] [A. Nicolson, K. K. Paliwal, Deep learning for minimum mean-square error approaches to speech enhancement, Speech Communication 111 (2019) 44 - 55, https://doi.org/10.1016/j.specom.2019.06.002.](https://maxwell.ict.griffith.edu.au/spl/publications/papers/spcom19_aaron_deep.pdf)
-->

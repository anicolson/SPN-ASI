<!--- 

export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

-->
Sum-Product Networks for Robust Automatic Speaker Recognition
====

Sum-product networks with Gaussuan leaves are used here as speaker models for automatic speaker recognition. An example of an SPN with univariate Gaussian leaves is shown in Figure 1. Marginalisation and bounded marginalisation, as proposed by Cook *et al.* ([link](https://doi.org/10.1016/S0167-6393(00)00034-0)), is used here to significantly increase the robustness of the SPN speaker models to noise. To identify the reliable spectral components for marginalisation, an *a priori* SNR estimator is used.

|![](./spk_model.jpg "SPN speaker model.")|
|----|
| <p align="center"> <b>Figure 1:</b> <a> SPN speaker model.</a> </p> |

SPN speaker models in [**SPFlow**](https://github.com/SPFlow/SPFlow)
====
The SPN speaker models are implemented in [**SPFlow**](https://github.com/SPFlow/SPFlow) version 0.0.4, please check out the SPFlow repository [here](https://github.com/SPFlow/SPFlow), and star their repository. The SPFlow library is modified to include bounded marginalisation, with the main modification in *spn.structure.leaves.parametric.gaussian_likelihood* as follows:

```
def gaussian_likelihood(node, data=None, dtype=np.float64, bmarg=None, ibm=None):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)
    scipy_obj, params = get_scipy_obj_params(node)
    if bmarg:
        ibm = ibm[:, node.scope]
        probs_reliable = np.expand_dims(scipy_obj.pdf(observations, **params), axis=1)
        probs_unreliable = np.expand_dims(scipy.stats.norm.cdf(observations, loc=params['loc'], scale=params['scale']), axis=1)
        probs = np.where(ibm, probs_reliable, probs_unreliable)
    else:
        probs[~marg_ids] = scipy_obj.pdf(observations, **params)
    return probs

```
In the latest version of SPFLow (0.0.34), this function has been changed to *spn.structure.leaves.parametric.continuous_likelihood*. [*Log spectral subband energies* (LSSEs)](https://maxwell.ict.griffith.edu.au/spl/publications/papers/icsps17_aaron.pdf) are used as the input feature to each of the SPN speaker models. 

IBM estimation using [**Deep Xi**](https://github.com/anicolson/DeepXi)
====
[**Deep Xi**](https://github.com/anicolson/DeepXi) is a deep learning approach to *a priori* SNR estimation, as described [here](https://doi.org/10.1016/j.specom.2019.06.002). A threshold of 0 dB is applied to the *a priori* SNR estimate given by Deep Xi, to give the IBM estimate. The IBM estimate is then used to identify the reliable LSSEs.

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

TIMIT corpus
-----
Place the TIMIT corpus in the data path. E.g. the train directory for TIMIT should be located at data/timit/train. Similarly, the test directory for TIMIT should be located at data/timit/test.

Noisy speech
-----
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

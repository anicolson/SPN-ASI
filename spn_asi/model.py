## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from joblib import Parallel, delayed
from scipy.stats import norm
from spn.algorithms.Inference import add_node_likelihood, \
	leaf_marginalized_likelihood,log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.utils import get_scipy_obj_params
from spn_asi.sig import SubbandFeatures
from spn_asi.utils import read_mat, read_wav, save_mat
from tqdm import tqdm, trange
import multiprocessing, os, pickle, sys
import numpy as np
import tensorflow as tf

def continuous_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
	"""
	Modified Gaussian log-likelihood function with marginalisation and bounded
	marginalisation for the leaves.

	Argument/s:
		node - defined in spflow documentation.
		data - defined in spflow documentation.
		dtype - defined in spflow documentation.
		kwargs - keyword arguments where bounds flag and IBM is passed.

	Returns:
		probs - defined in spflow documentation.
	"""
	bounds = kwargs['bounds']
	ibm = kwargs['ibm']
	probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype, log_space=True)
	scipy_obj, params = get_scipy_obj_params(node)
	if bounds:
		ibm = ibm[:, node.scope]
		probs_reliable = np.expand_dims(scipy_obj.logpdf(observations, **params), axis=1)
		probs_unreliable = np.expand_dims(norm.logcdf(observations, loc=params['loc'], scale=params['scale']), axis=1)
		probs = np.where(ibm, probs_reliable, probs_unreliable)
	else: probs[~marg_ids] = scipy_obj.logpdf(observations, **params)
	return probs

add_node_likelihood(Gaussian, log_lambda_func=continuous_log_likelihood)

class SPNASISystem():
	"""
	Sum-product network automatic speaker identification system.
	"""
	def __init__(
		self,
		N_d,
		N_s,
		K,
		f_s,
		M,
		spk_list,
		ver,
		**kwargs
		):
		"""
		Argument/s:
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			M - number of filters.
			spk_list - list of speakers.
			ver - system version.
		"""
		self.spk_list = spk_list
		self.ver = ver
		self.n_spk = len(self.spk_list)
		self.feat = SubbandFeatures(N_d, N_s, K, f_s, M)

	def train(
			self,
			spk_obs,
			model_path,
			min_instances_slice,
			threshold,
		):
		"""
		Speaker model training.

		Argument/s:
			spk_obs - observations for each speaker.
			model_path - path to speaker models.
			min_instances_slice - minimum number of instances to slice.
			threshold - threshold.
		"""

		ncores = multiprocessing.cpu_count()

		print('Training...')
		for i, j in enumerate(self.spk_list):

			print(chr(27) + "[2J")
			print("Training speaker: %i/%i (%s)..." % (i+1, self.n_spk, j))

			max_len = max(spk_obs[j]['train_x']['wav_len']) # find maximum sequence length.
			train_x_wav = np.zeros([len(spk_obs[j]['train_x']['wav_path']), max_len], np.int16)
			for k, l in enumerate(spk_obs[j]['train_x']['wav_path']):
				wav, _ = read_wav(l)
				wav_len = spk_obs[j]['train_x']['wav_len'][k]
				train_x_wav[k,:wav_len] = wav

			train_x = self.feat.observation(train_x_wav,
				spk_obs[j]['train_x']['wav_len']).numpy()

			print("Features extracted.")

			ds_context = Context(parametric_types=[Gaussian]*self.feat.M).add_domains(train_x)
			with Silence():
				spn_spk = learn_parametric(
					train_x,
					ds_context,
					min_instances_slice=min_instances_slice,
					threshold=threshold,
					cpus=ncores
					)

			spk_path = model_path + '/' + i + '.p' # speaker model save path.
			with open(spk_path, 'wb') as f: pickle.dump(spn_spk, f)

	def identification(
		self,
		spk_obs,
		model_path,
		marg,
		bounds,
		n_workers,
		eval_cond,
		xi_hat_path=None,
		test_set_name="",
		):
		"""
		Automatic speaker identification.

		Argument/s:
			spk_obs - list of training examples for each speaker.
			model_path - path to speaker models.
			marg - marginalisation flag.
			bounds - bounds flag.
			n_workers - number of workers for Parallel().
			eval_cond - evaluate different SNR level and noise source
				conditions.
			xi_hat_path - path to a priori SNR estimate .mat files used to
				compute the ideal binary mask (IBM) estimates.
			test_set_name - name of test set.
		"""

		ibm_hat = None

		## LOAD SPEAKER MODELS
		spk_models = {}
		print("Loading speaker models...")
		for i in tqdm(self.spk_list):
			with open(model_path + '/' + i + '.p', 'rb') as f:
				spk_models[i] = pickle.load(f)

		correct = 0; total = 0
		results = {}
		t = trange(self.n_spk, desc='Acc=0%', leave=True)
		for i in t:
			for j in range(len(spk_obs[self.spk_list[i]]['test_x']['wav_path'])):

				tgt_spk = self.spk_list[i]
				wav_path = spk_obs[tgt_spk]['test_x']['wav_path'][j]
				wav_len = spk_obs[tgt_spk]['test_x']['wav_len'][j]

				wav, _ = read_wav(wav_path)
				test_x = self.feat.observation(wav, wav_len).numpy()

				if marg:
					xi_hat = read_mat(xi_hat_path + '/' + wav_path.split('/')[-1][:-4])
					xi_hat = xi_hat['xi_hat']
					ibm_hat = tf.greater(tf.linalg.matmul(xi_hat,
						self.feat.H, transpose_b=True), 1.0).numpy()

				if marg and not bounds: test_x = np.where(ibm_hat, test_x, np.full_like(test_x, np.nan))

				sll = np.array(Parallel(n_jobs=n_workers)(delayed(self.sequence_log_likelihood)(spk_models[self.spk_list[l]],
				 	test_x, self.spk_list[l], bounds=bounds, ibm=ibm_hat) for l in range(self.n_spk)))

				if sll[np.argmax(sll[:,0].astype(np.float32)), 1] == tgt_spk:
					correct_identification = 1
				else: correct_identification = 0

				correct = correct + correct_identification
				total = total + 1

				if eval_cond:
					noise_src = spk_obs[tgt_spk]['test_x']['noise_src'][j]
					snr_level = spk_obs[tgt_spk]['test_x']['snr'][j]
					results = self.add_score(results, (noise_src, snr_level), correct_identification)

				t.set_description("Acc={:3.2f}%".format(100*(correct/total)))
				t.refresh()

		results_path = "results/" + test_set_name
		if not os.path.exists(results_path): os.makedirs(results_path)

		if not os.path.exists(results_path + "/average.csv"):
			with open(results_path + "/average.csv", "w") as f:
				f.write("ver,acc\n")

		with open(results_path + "/average.csv", "a") as f:
			f.write("{:s},{:.4f}\n".format(self.ver, 100*(correct/total)))

		if eval_cond:
			noise_srcs, snr_levels = set(), set()
			for key, value in results.items():
				noise_srcs.add(key[0])
				snr_levels.add(key[1])

			if not os.path.exists(results_path + "/" + self.ver + ".csv"):
				with open(results_path + "/" + self.ver + ".csv", "w") as f:
					f.write("marg,bounds,noise,snr_db,acc\n")

			with open(results_path + "/" + self.ver + ".csv", "a") as f:
				for i in sorted(noise_srcs):
					for j in sorted(snr_levels):
						f.write("{},{},{},{},{:.2f}\n".format(marg,
							bounds, i, j, 100*np.mean(results[(i,j)])))

	def sequence_log_likelihood(self, model, observation, spk, bounds=False, ibm=None):
		"""
		Compute sequence log-likelihood for over observation.

		Argument/s:
			model - SPN speaker model.
			observation - LSSE observation.
			spk - speaker.
			bounds - bounds flag.
			ibm - ideal binary mask (IBM).

		Returns:
			sll - sequence log-likelihood.
			spk - speaker.
		"""
		sll = np.sum(log_likelihood(model, observation, bounds=bounds, ibm=ibm))
		return sll, spk

	def add_score(self, dict, key, score):
		"""
		Adds score/s to the list for the given key.

		Argument/s:
			dict - dictionary with condition as keys and a list of objective
				scores as values.
			key - noisy-speech conditions.
			score - objective score.

		Returns:
			dict - updated dictionary.
		"""
		if isinstance(score, list):
			if key in dict.keys(): dict[key].extend(score)
			else: dict[key] = score
		else:
			if key in dict.keys(): dict[key].append(score)
			else: dict[key] = [score]
		return dict

class Silence(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

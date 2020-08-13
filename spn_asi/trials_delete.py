## FILE:           trials.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Function for automatic speaker verification.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from datetime import datetime
from dev.datasets.SITW import read_core_core_list
from dev.filterbank import melfbank_tapered
from dev.utils import read_wav, read_mat
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.stats import norm
from spn.algorithms.Inference import leaf_marginalized_log_likelihood, log_likelihood
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.utils import get_scipy_obj_params
from tqdm import trange
import dev.se_batch as batch
import numpy as np
import multiprocessing, os, pickle, sys
import tensorflow as tf

import dev.utils as utils

def gaussian_log_likelihood(node, data=None, dtype=np.float64, bounds=None, ibm=None):
	probs, marg_ids, observations = leaf_marginalized_log_likelihood(node, data, dtype)
	scipy_obj, params = get_scipy_obj_params(node)
	if bounds:
		ibm = ibm[:, node.scope]
		probs_reliable = np.expand_dims(scipy_obj.logpdf(observations, **params), axis=1)
		probs_unreliable = np.expand_dims(norm.logcdf(observations, loc=params['loc'], scale=params['scale']), axis=1)
		probs = np.where(ibm, probs_reliable, probs_unreliable)
	else: probs[~marg_ids] = scipy_obj.logpdf(observations, **params)
	return probs

from spn.algorithms.Inference import add_node_likelihood
add_node_likelihood(Gaussian, log_lambda_func=gaussian_log_likelihood)

def sequence_log_likelihood(spn, observations, spk_id, bounds=False, ibm=None):
	sll = np.sum(log_likelihood(spn, observations, bounds=bounds, ibm=ibm))
	return sll, spk_id

def identification(sess, args):
	if not hasattr(args, 'ncores'): args.ncores = multiprocessing.cpu_count()
	H = melfbank_tapered(args.M, args.NFFT/2 + 1, args.f_s)
	ibm_hat = None

	## LOAD SPEAKER MODELS
	spk_models = []
	for i in range(len(args.num2id)):
		with open(args.model_path + '/' + args.num2id[i] + '.p', 'rb') as f:
			spk_models.append({'spk_num': i, 'spk_id': args.num2id[i], 'spk_model': pickle.load(f)})

	correct = 0; total = 0
	results = {}
	t = trange(len(args.test_list), desc='Acc=0%', leave=True)
	for j in t:
		i = args.test_list[j]
		file_info = (i["file_path"].rsplit('/', 1)[1].rsplit('.', 1)[0].split('_'))

		if file_info[2] in results:
			if file_info[3] in results[file_info[2]]: results[file_info[2]][file_info[3]]['total'] += 1
			else:
				results[file_info[2]][file_info[3]] = {}
				results[file_info[2]][file_info[3]]['total'] = 1
				results[file_info[2]][file_info[3]]['correct'] = 0
		else:
			results[file_info[2]] = {}
			results[file_info[2]][file_info[3]] = {}
			results[file_info[2]][file_info[3]]['total'] = 1
			results[file_info[2]][file_info[3]]['correct'] = 0


		(wav, _) = read_wav(i['file_path']) # read wav from given file path.

		lsse = sess.run(args.feat, feed_dict={args.s_ph: [wav], args.s_len_ph: [i['seq_len']]}) # mini-batch.

		if args.marg:
			xi_hat = read_mat(args.ibm_hat_path + '/' + i['file_path'].rsplit('/', 1)[1][:-4])
			xi_hat = xi_hat['xi_hat']
			ibm_hat = np.greater(np.matmul(xi_hat, np.transpose(H)), 1.0)

		if args.marg and not args.bounds: lsse = np.where(ibm_hat, lsse, np.full_like(lsse, np.nan))

		sll = np.array(Parallel(n_jobs=args.ncores)(delayed(sequence_log_likelihood)(spk_models[j]['spk_model'],
			lsse, spk_models[j]['spk_id'], bounds=args.bounds, ibm=ibm_hat) for j in range(args.num_spk)))

		if sll[np.argmax(sll[:,0].astype(np.float32)),1] == file_info[0]:
			results[file_info[2]][file_info[3]]['correct'] += 1

		all_total = 0
		all_correct = 0
		for a in results:
			for b in results[a]:
				all_total += results[a][b]['total']
				all_correct += results[a][b]['correct']
		t.set_description("Acc={:3.2f}%".format(100*(all_correct/all_total)))
		t.refresh()

	for a in results:
		for b in results[a]:
			print(a, b, 100*(results[a][b]['correct']/results[a][b]['total']))

	print('Complete.')

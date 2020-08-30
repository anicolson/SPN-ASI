## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os, sys
from spn_asi.args import get_args
from spn_asi.model import SPNASISystem
from spn_asi.dataset import timit_dataset, noisy_speech_dataset
from spn_asi.utils import gpu_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

	## GET COMMAND LINE ARGUMENTS
	args = get_args()

	## GPU CONFIGURATION
	config = gpu_config(args.gpu)

	## TRAINING AND TESTING SET ARGUMENTS
	args.model_path = args.model_path + '/' + args.ver
	if not os.path.exists(args.data_path): os.makedirs(args.data_path) # make data path directory.
	if not os.path.exists(args.model_path): os.makedirs(args.model_path) # make model path directory.
	N_d = int(args.f_s*args.T_d*0.001) # window length (samples).
	N_s = int(args.f_s*args.T_s*0.001) # window shift (samples).
	K = N_d # number of DFT components.

	## TIMIT DATASET
	spk_list, spk_obs = timit_dataset(args.set_path)

	## NOISY SPEECH DATASET
	_, noisy_spk_obs = noisy_speech_dataset(args.noisy_speech_path)

	## ASI SYSTEM
	asi = SPNASISystem(N_d, N_s, K, args.f_s, args.n_subbands, spk_list, args.ver)

	## TRAIN SPEAKER MODELS
	if args.train:
		asi.train(
			spk_obs=spk_obs,
			model_path=args.model_path,
			min_instances_slice=args.min_instances_slice,
			threshold=args.threshold,
		)

	## AUTOMATIC SPEAKER IDENTIFICATION
	if args.identification:

		## CLEAN SPEECH
		# asi.identification(
		# 	spk_obs=spk_obs,
		# 	model_path=args.model_path,
		# 	marg=False,
		# 	bounds=False,
		# 	n_workers=args.n_workers,
		# 	eval_cond=False,
		# 	test_set_name = "timit",
		# )

		## NOISY SPEECH
		asi.identification(
			spk_obs=noisy_spk_obs,
			model_path=args.model_path,
			marg=args.marg,
			bounds=args.bounds,
			n_workers=args.n_workers,
			eval_cond=True,
			xi_hat_path=args.xi_hat_path,
			test_set_name = "noisy_speech",
		)

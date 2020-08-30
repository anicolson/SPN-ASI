## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import math

def read_dtype(x):
	x = x.replace("neg_", "-")
	if x == 'pi': return math.pi
	elif x == '-pi': return -math.pi
	elif any(map(str.isdigit, x)):
		if '.' in x: return float(x)
		else: return int(x)
	else:
		return x

def str_to_list(x):
	if ';' in x: return [[read_dtype(z) for z in y.split(',')] for y in x.split(';')]
	elif ',' in x: return [read_dtype(y) for y in x.split(',')]
	else: return read_dtype(x)

def str_to_bool(s): return s.lower() in ("yes", "true", "t", "1")

def get_args():
	parser = argparse.ArgumentParser()

	## OPTIONS (GENERAL)
	parser.add_argument('--gpu', default='0', type=str, help='GPU selection')
	parser.add_argument('--ver', type=str, help='Model version')
	parser.add_argument('--train', default=False, type=str_to_bool, help='Perform training')
	parser.add_argument('--identification', default=False, type=str_to_bool, help='Perform identification')
	parser.add_argument('--n_workers', type=int, help='Number of workers for Parallel().')

	## PATHS
	parser.add_argument('--model_path', default='model', type=str, help='Model save path')
	parser.add_argument('--set_path', default='set', type=str, help='Path to datasets')
	parser.add_argument('--data_path', default='data', type=str, help='Save data path')
	parser.add_argument('--noisy_speech_path', type=str, help='Path to noisy speech files')
	parser.add_argument('--xi_hat_path', type=str, help='Path to a priori SNR estimate .mat files for ideal binary mask (IBM) estimates.')

	## FEATURES
	parser.add_argument('--f_s', type=int, help='Sampling frequency (Hz)')
	parser.add_argument('--T_d', type=int, help='Window duration (ms)')
	parser.add_argument('--T_s', type=int, help='Window shift (ms)')
	parser.add_argument('--n_subbands', type=int, help='Number of subbands for filterbank')

	## MFT
	parser.add_argument('--marg', default=False, type=str_to_bool, help='Use marginalisation')
	parser.add_argument('--bounds', default=False, type=str_to_bool, help='Use bounds for integration')

	## SPN
	parser.add_argument('--min_instances_slice', type=int, help='Minimum number of instances to slice')
	parser.add_argument('--threshold', type=float, help='Threshold')

	args = parser.parse_args()
	return args

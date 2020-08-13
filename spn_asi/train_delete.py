## FILE:           train.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Training function for automatic speaker identification.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from datetime import datetime
from dev.speaker.identification.batch import batch
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Gaussian
from tqdm import trange
import numpy as np
import multiprocessing, os, pickle, random, sys
import tensorflow as tf

class silence(object):
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

def train(sess, args):

	if not hasattr(args, 'ncores'): args.ncores = multiprocessing.cpu_count()
	for i in range(args.num_spk):
		spk_list = [item for item in args.train_list if item["spk_num"] == i]
		spn_path = args.model_path + '/' + spk_list[0]['spk_id'] + '.p'

		batch_sig, seq_len, spk_num = batch(spk_list)
		batch_x = sess.run(args.feat, feed_dict={args.s_ph: batch_sig, args.s_len_ph: seq_len}) # mini-batch.

		if not os.path.isfile(spn_path):
			with open(spn_path, 'wb') as f: pickle.dump([], f)
			print(chr(27) + "[2J")
			print("Learn structure, spk: %i (%s)... (min_instances_slice: %i, threshold: %1.3f)." % (i, 
				spk_list[0]['spk_id'], args.min_instances_slice, args.threshold))
			print("Features extracted.")
			ds_context = Context(parametric_types=[Gaussian]*args.M).add_domains(batch_x)
			with silence(): 
				spn_spk = learn_parametric(batch_x, ds_context, min_instances_slice=args.min_instances_slice, 
					threshold=args.threshold, cpus=args.ncores)
			with open(spn_path, 'wb') as f: pickle.dump(spn_spk, f)

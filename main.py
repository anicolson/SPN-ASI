# FILE:           spn.py
# DATE:           2019
# AUTHOR:         Aaron Nicolson
# AFFILIATION:    Signal Processing Laboratory, Griffith University
# BRIEF:          SPN speaker models.

from joblib import Parallel, delayed
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Gaussian
from os.path import expanduser
import spn_batch, featpy, multiprocessing, os, pickle, sys
import numpy as np
import tensorflow as tf
sys.path.insert(0, './DeepXi')
sys.path.insert(0, './DeepXi/lib')
import deepxi, utils
import scipy.special as spsp

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

## SEQUENCE LOG-LIKELIHOOD
def sequence_log_likelihood(spn, observations, spk_id, bmarg=False, ibm=None):
	sll = np.sum(log_likelihood(spn, observations, bmarg=bmarg, ibm=ibm))
	return sll, spk_id

## IBM ESTIMATE
def ibm_hat(sess, net, wav, wav_len, args):
	x_MS_3D_out = sess.run(net.x_MS_3D, feed_dict={net.x_ph: wav, 
	 	net.x_len_ph: wav_len}) 
	x_seq_len_out = sess.run(net.x_seq_len, feed_dict={net.x_len_ph: wav_len}) 
	mu_np = sess.run(net.mu); # mean tf.constant to np.array.
	sigma_np = sess.run(net.sigma); # standard deviation tf.constant to np.array.
	output_out = sess.run(net.output, feed_dict={net.x_MS_ph: x_MS_3D_out, net.x_MS_len_ph: x_seq_len_out, net.training_ph: False}) # output of network.
	output_out = utils.np_sigmoid(output_out)
	xi_dB_hat_out = np.add(np.multiply(np.multiply(sigma_np, np.sqrt(2.0)), spsp.erfinv(np.subtract(np.multiply(2.0, output_out), 1))), mu_np); # a priori SNR estimate.			
	xi_hat_out = np.power(10.0, np.divide(xi_dB_hat_out, 10.0))
	return np.greater(np.matmul(xi_hat_out[0:-1,:], np.transpose(args.H_tapered)), 1.0); # LSSE IBM estimate.

## TRAINING
def train(args):
	print('Training...')
	for i in range(len(args.spk_list)):
		spn_path = args.MODEL_DIR + '/' + args.spk_list[i]['spk_id'] + '.p'
		if not os.path.isfile(spn_path):
			with open(spn_path, 'wb') as f: pickle.dump([], f)
			print(chr(27) + "[2J")
			print("Learn structure, spk: %i (%s)... (min_instances_slice: %i, threshold: %1.3f)." % (i, 
				args.spk_list[i]['spk_id'], args.min_instances_slice, args.threshold))
			train_batch = featpy.lsse(args.spk_list[i]['train_clean_speech'], 
				args.spk_list[i]['train_clean_speech_len'], args.Nw, args.Ns, args.NFFT, args.fs, args.H)
			print("Features extracted.")
			ds_context = Context(parametric_types=[Gaussian]*args.M).add_domains(train_batch)
			with silence(): 
				spn_spk = learn_parametric(train_batch, ds_context, min_instances_slice=args.min_instances_slice, 
					threshold=args.threshold, cpus=args.ncores)
			with open(spn_path, 'wb') as f: pickle.dump(spn_spk, f)

## CLEAN SPEECH TESTING
def test_clean_speech(args):
	print('Clean speech testing...')
	correct = 0; total = 0
	test_size = len(args.spk_list)
	for i in range(test_size):
		with open(args.MODEL_DIR + '/' + args.spk_list[i]['spk_id'] + '.p', 'rb') as f:
			args.spk_list[i]['spn'] = pickle.load(f)
	for i in range(test_size):
		print(chr(27) + '[2Jtest clean speech,\nspeaker: %i,\ncorrect: %i,\ntotal: %i,\nmin_instances_slice: %i.' % 
			(i, correct, total, args.min_instances_slice))
		sa1 = featpy.lsse(args.spk_list[i]['sa1'], args.spk_list[i]['sa1_len'], args.Nw, args.Ns, args.NFFT, args.fs, args.H)
		sa2 = featpy.lsse(args.spk_list[i]['sa2'], args.spk_list[i]['sa2_len'], args.Nw, args.Ns, args.NFFT, args.fs, args.H)
		sll_sa1 = np.array(Parallel(n_jobs=args.ncores)(delayed(sequence_log_likelihood)(args.spk_list[j]['spn'], 
			sa1, args.spk_list[j]['spk_id']) for j in range(test_size)))
		sll_sa2 = np.array(Parallel(n_jobs=args.ncores)(delayed(sequence_log_likelihood)(args.spk_list[j]['spn'], 
			sa2, args.spk_list[j]['spk_id']) for j in range(test_size)))
		if sll_sa1[np.argmax(sll_sa1[:,0].astype(np.float32)),1] in args.spk_list[i]['spk_id']:
		 	correct += 1
		if sll_sa2[np.argmax(sll_sa2[:,0].astype(np.float32)),1] in args.spk_list[i]['spk_id']:
			correct += 1
		total += 2
	print("\naccuracy: %3.2f%%\ncorrect: %i,\ntotal: %i.\n" % (100*(correct/total), correct, total))
	with open("results.txt", "a") as f:
		f.write("Clean speech: acc=%3.2f%%, corr=%i, tot=%i, ver=%s.\n" % (100*(correct/total), correct, total, args.ver))

## NOISY SPEECH TESTING
def test_noisy_speech(sess=None, net=None, args=None):
	print('Testing on noisy speech...')
	test_size = len(args.spk_list)
	IBM_hat = None
	if args.mft == 'bmarg': bmarg_flag = True
	else: bmarg_flag = False
	for i in range(test_size):
		with open(args.MODEL_DIR + '/' + args.spk_list[i]['spk_id'] + '.p', 'rb') as f:
			args.spk_list[i]['spn'] = pickle.load(f)
	for i in range(len(args.noise_src)):
		for q in range(len(args.snr)):
			correct = 0; total = 0
			for k in range(test_size):
				for j in ['_sa1_', '_sa2_']:
					noisy_speech_file = args.NOISY_SPEECH_DIR + '/' + args.spk_list[k]['spk_id'] + j + args.noise_src[i] + '_' + args.snr[q] + '.wav'
					if os.path.isfile(args.NOISY_SPEECH_DIR + '/' + args.spk_list[k]['spk_id'] + j + 
							args.noise_src[i] + '_' + args.snr[q] + '.wav'):
						wav, wav_len, _, _ = spn_batch._batch(args.NOISY_SPEECH_DIR, noisy_speech_file, [])
						lsse = featpy.lsse(wav, wav_len, args.Nw, args.Ns, args.NFFT, args.fs, args.H)
						if args.mft == 'marg' or args.mft == 'bmarg': IBM_hat = ibm_hat(sess, net, wav, wav_len, args)
						if args.mft == 'marg': lsse = np.where(IBM_hat, lsse, np.full_like(lsse, np.nan))
						sll = np.array(Parallel(n_jobs=args.ncores)(delayed(sequence_log_likelihood)(args.spk_list[j]['spn'], 
							lsse, args.spk_list[j]['spk_id'], bmarg=bmarg_flag, ibm=IBM_hat) for j in range(test_size)))
						total += 1
						if sll[np.argmax(sll[:,0].astype(np.float32)),1] in args.spk_list[k]['spk_id']: correct += 1
					print("corr=%i, total=%i, noise=%s, SNR=%s, mft=%s, ver=%s." % (correct, total, args.noise_src[i], args.snr[q], args.mft, args.ver), end="\r")
				print("\nAccuracy: %3.2f%%." % (100*(correct/total)))
				with open("results.txt", "a") as f:
					f.write("%s@%s: acc=%3.2f%%, corr=%i, tot=%i, mft=%s, ver=%s.\n" % (args.noise_src[i], args.snr[q], 100*(correct/total), 
					correct, total, args.mft, args.ver))
	print('Complete.')

## ADDITIONAL ARGUMENTS
def add_args(args):
	if not os.path.exists(args.DATA_DIR): os.makedirs(args.DATA_DIR) # make data path directory.
	if not os.path.exists(args.MODEL_DIR): os.makedirs(args.MODEL_DIR) # make model path directory.
	args.snr = ['-5dB', '0dB', '5dB', '10dB', '15dB']
	args.noise_src = ['voice-babble', 'street-music-26270', 'f16', 'factory-welding']
	args.Nw = int(args.fs*args.Tw*0.001) # window length (samples).
	args.Ns = int(args.fs*args.Ts*0.001) # window shift (samples).
	args.NFFT = int(pow(2, np.ceil(np.log2(args.Nw)))) # number of DFT components.

	args.M = 26 # number of filters for filterbank.
	args.H = featpy.melfbank(args.M, args.NFFT/2 + 1, args.fs)
	args.H_tapered = featpy.melfbank_tapered(args.M, args.NFFT/2 + 1, args.fs)
	
	if not hasattr(args, 'ncores'): args.ncores = multiprocessing.cpu_count()

	## SPEAKER LIST
	if os.path.exists(args.DATA_DIR + '/spk_list.p'):
		print('Loading speaker list from pickle file...')
		with open(args.DATA_DIR + '/spk_list.p', 'rb') as f:
			args.spk_list = pickle.load(f)
	else:
		print('Creating speaker list, as no pickle file exists...')
		args.spk_list = [] # speaker list.
		for path, _, _ in os.walk(args.TIMIT_DIR):
			if path[-5] in ('m', 'f'):
				train_clean_speech, train_clean_speech_len, _, _ = spn_batch._batch(path, ['si*.wav', 'sx*.wav'] , []) # clean training waveforms and lengths.
				sa1, sa1_len, _, _ = spn_batch._batch(path, ['sa1.wav'] , []) # sa1 clean testing waveform and length.
				sa2, sa2_len, _, _ = spn_batch._batch(path, ['sa2.wav'] , []) # sa2 clean testing waveform and length.
				args.spk_list.append({'path': path, 'spk_id': path.rsplit('/', 1)[-1], 
					'train_clean_speech': train_clean_speech, 'train_clean_speech_len': train_clean_speech_len, 
					'sa1': sa1, 'sa1_len': sa1_len, 'sa2': sa2, 'sa2_len': sa2_len}) # append dictionary.
		with open(args.DATA_DIR + '/spk_list.p', 'wb') as f: 		
			pickle.dump(args.spk_list, f)
	print('%i total speakers.' % (len(args.spk_list)))
	return args

if __name__ == '__main__':

	## ARGUMENTS
	spn_args = utils.args()
	spn_args.DATA_DIR = '/data/SPN-Spk-Rec'
	spn_args.TIMIT_DIR = expanduser("~") + '/data/timit'
	spn_args.NOISY_SPEECH_DIR = expanduser("~") + '/data/tmp/noisy_speech'
	spn_args.MODEL_DIR = 'model/' + spn_args.ver
	spn_args = add_args(spn_args)

	## TRAINING
	if spn_args.train: train(spn_args)

	## TESTING
	if spn_args.test_clean_speech: test_clean_speech(spn_args)
	if spn_args.test_noisy_speech: 

		## DEEP XI FOR IBM ESTIMATION
		deepxi_args = utils.args()
		deepxi_args.ver = '3a'
		deepxi_args.blocks = ['C3'] + ['B5']*40 + ['O1']
		deepxi_args.epoch = 175
		deepxi_args.stats_path = './DeepXi/stats'
		deepxi_args.model_path = './DeepXi/model'
		deepxi_args.train = False
		deepxi_args = deepxi.deepxi_args(deepxi_args)
		deepxi_args.infer = True
		deepxi_graph = tf.Graph()
		with deepxi_graph.as_default():
			deepxi_net = deepxi.deepxi_net(deepxi_args)
		config = utils.gpu_config(deepxi_args.gpu)
		deepxi_sess = tf.Session(config=config, graph=deepxi_graph)
		deepxi_net.saver.restore(deepxi_sess, deepxi_args.model_path + '/epoch-' + str(deepxi_args.epoch)) # load model for epoch.

		## NO MARGINALISATION
		spn_args.mft = None
		test_noisy_speech(None, None, spn_args)

		## MARGINALISATION
		spn_args.mft = 'marg'
		test_noisy_speech(deepxi_sess, deepxi_net, spn_args)

		## BOUNDED MARGINALISATION
		spn_args.mft = 'bmarg'
		test_noisy_speech(deepxi_sess, deepxi_net, spn_args)

		# CLOSE TF GRAPH
		deepxi_sess.close()

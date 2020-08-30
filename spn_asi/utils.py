## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from scipy.io import loadmat, savemat
from soundfile import SoundFile, SEEK_END
import numpy as np
import glob, os, pickle, platform
import soundfile as sf
import tensorflow as tf

def save_wav(path, wav, f_s):
	"""
	Save .wav file.

	Argument/s:
		path - absolute path to save .wav file.
		wav - waveform to be saved.
		f_s - sampling frequency.
	"""
	wav = np.squeeze(wav)
	if isinstance(wav[0], np.float32): wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
	sf.write(path, wav, f_s)

def read_wav(path):
	"""
	Read .wav file.

	Argument/s:
		path - absolute path to save .wav file.

	Returns:
		wav - waveform.
		f_s - sampling frequency.
	"""
	wav, f_s = sf.read(path, dtype='int16')
	return wav, f_s

def save_mat(path, data, name):
	"""
	Save .mat file.

	Argument/s:
		path - absolute path to save .mat file.
		data - data to be saved.
		name - dictionary key name.
	"""
	if not path.endswith('.mat'): path = path + '.mat'
	savemat(path, {name: data})

def read_mat(path):
	"""
	Read .mat file.

	Argument/s:
		path - absolute path to save .mat file.

	Returns:
		Dictionary.
	"""
	if not path.endswith('.mat'): path = path + '.mat'
	return loadmat(path)

def gpu_config(gpu_selection, log_device_placement=False):
	"""
	Selects GPU.

	Argument/s:
		gpu_selection - GPU to use.
		log_device_placement - log the device that each node is placed on.
	"""
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selection)
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

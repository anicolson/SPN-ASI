## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import glob, os
from soundfile import SoundFile, SEEK_END

def timit_dataset(timit_path):
	"""
	For the TIMIT corpus detirmines the speakers and their training and test observations.
	For training, the si* and sx* sets are used, for testing, the sa* set is
	used.

	Argument/s:
		timit_path - path to the TIMIT corpus.

	Returns:
		spk_list - list of speakers.
		spk_obs - observations for each speaker.
	"""
	## SPEAKER LIST
	print('Finding speakers and speaker observations...')
	spk_list = []
	spk_obs = {} # speaker list.
	id_count = 0
	for i, _, _ in os.walk(timit_path):
		if i[-5] in ('m', 'f'):
			for j in glob.glob(os.path.join(i, '*.wav')):
				f = SoundFile(j)
				wav_len = f.seek(0, SEEK_END)
				if wav_len == -1:
					wav, _ = read_wav(i)
					wav_len = len(wav)
				spk = i.split('/')[-1]
				if spk not in spk_obs:
					spk_obs[spk] = {'spk_id': id_count,
						'train_x': {'wav_path': [], 'wav_len': []},
						'test_x': {'wav_path': [], 'wav_len': []},
						}
					id_count = id_count + 1
					spk_list.append(spk)
				timit_set = j.split('/')[-1][0:2]
				if timit_set in ['si', 'sx']:
					spk_obs[spk]['train_x']['wav_path'].append(j)
					spk_obs[spk]['train_x']['wav_len'].append(wav_len)
				if timit_set == 'sa':
					spk_obs[spk]['test_x']['wav_path'].append(j)
					spk_obs[spk]['test_x']['wav_len'].append(wav_len)
			print('%i total speakers.' % (id_count), end="\r")
	return spk_list, spk_obs

def noisy_speech_dataset(noisy_speech_path):
	"""
	Detirmines the noisy speech test observations.

	Argument/s:
		noisy_speech_path - path to the noisy_speech .wav files.

	Returns:
		spk_list - list of speakers.
		spk_obs - observations for each speaker.
	"""
	## SPEAKER LIST
	print('Finding noisy speaker observations...')
	spk_list = []
	spk_obs = {} # speaker list.
	id_count = 0
	for i in glob.glob(os.path.join(noisy_speech_path, '*.wav')):
		f = SoundFile(i)
		wav_len = f.seek(0, SEEK_END)
		if wav_len == -1:
			wav, _ = read_wav(i)
			wav_len = len(wav)
		spk = i.split('/')[-1].split('_')[0]
		if spk not in spk_obs:
			spk_obs[spk] = {'spk_id': id_count,
				'test_x': {'wav_path': [], 'wav_len': [],
				'noise_src': [], 'snr': []},
				}
			id_count = id_count + 1
			spk_list.append(spk)

		noise_src = i.split('/')[-1].split('_')[2]
		snr = int(i.split('/')[-1].split('_')[3][:-6])

		spk_obs[spk]['test_x']['wav_path'].append(i)
		spk_obs[spk]['test_x']['wav_len'].append(wav_len)
		spk_obs[spk]['test_x']['noise_src'].append(noise_src)
		spk_obs[spk]['test_x']['snr'].append(snr)

	return spk_list, spk_obs

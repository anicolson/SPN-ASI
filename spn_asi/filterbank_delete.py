## FILE:           filterbank.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Compute features from filterbank representation.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import tensorflow as tf
from sig.feat.tf1 import polar

def mfcc(x, x_len, N_w, N_s, NFFT, f_s, M):
	'''
	Computes mel-frequency cepstral coefficients.

	Input/s: 
		x - waveform.
		x_len - waveform length.
		Nw - window length (samples).
		Ns - window shift (samples).
		NFFT - number of DFT bins.
		M - number of filters for mel-scale filterbank.

	Output/s: 
		MFCC - mel-frequency cepstral coefficients.
		L - number of frames.
	'''
	LSSE, L = lsse(x, x_len, N_w, N_s, NFFT, f_s, M)
	return tf.signal.dct(LSSE, type=2), L
	# return LSSE, L
	# return lifter(tf.signal.dct(LSSE, type=2), M), L

# def lifter(cepstra, L=22):
#     """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
#     magnitude of the high frequency DCT coeffs.
#     :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
#     :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
#     """
#     if L > 0:
#         nframes,ncoeff = np.shape(cepstra)
#         n = np.arange(ncoeff)
#         lift = 1 + (L/2.)*np.sin(np.pi*n/L)
#         return lift*cepstra
#     else:
#         # values of L <= 0, do nothing
#         return cepstra

def lifter(c, M):
	'''
	Cepstra liftering.

	Input/s:
		c - cepstra coefficients.
		M - number of filters for mel-scale filterbank.
	
	Output/s:
		c - liftered cepstra coefficients.
	'''

	lifter = 1.0 + (M/2)*np.sin(np.pi*np.arange(0.0, M, 1.0, dtype=np.float32)/M) # lifter.
	return tf.multiply(c, lifter) # liftering.

def lsse(x, x_len, N_w, N_s, NFFT, f_s, M):
	'''
	Computes log-spectral subband energies (LSSE) using a mel-scale filterbank.

	Input/s: 
		x - waveform.
		x_len - waveform length.
		Nw - window length (samples).
		Ns - window shift (samples).
		NFFT - number of DFT bins.
		M - number of filters for mel-scale filterbank.

	Output/s: 
		LSSE - log spectral subband energies.
	'''
	H = melfbank(M, int(NFFT/2 + 1), f_s)
	MAG, L, _ = polar.input(x, x_len, N_w, N_s, NFFT, f_s)
	LSSE = tf.log(tf.maximum(tf.matmul(tf.square(MAG), tf.transpose(H)), 1e-12))
	mask = tf.cast(tf.expand_dims(tf.sequence_mask(L), 2), tf.float32)
	return tf.multiply(LSSE, mask), L
	
def hz2mel(f):
	'''
	Converts a value from the Hz scale to a value in the Mel scale.
		
	Input: 
		f - Hertz value.
		
	Output: 
		m - mel value.
	'''
	return 2595*np.log10(1 + (f/700))

def mel2hz(m):
	'''
	converts a value from the mel scale to a value in the Hz scale.
		
	Input: 
		m - mel value.
		
	Output: 
		f - Hertz value.
	'''
	return 700*((10**(m/2595)) - 1)

def bpoint(m, M, NFFT, fs, fl, fh):
	'''
	detirmines the frequency bin boundary point for a filterbank.
	
	Inputs:
		m - filterbank.
		M - total filterbanks.
		NFFT - number of frequency bins.
		fs - sampling frequency.
		fl - lowest frequency.
		fh - highest frequency.

	Output:
		f - frequency bin boundary point.
	'''
	return ((2*NFFT)/fs)*mel2hz(hz2mel(fl) + m*((hz2mel(fh) - hz2mel(fl))/(M + 1))) # boundary point.
	
def melfbank(M, NFFT, fs):
	'''
	creates triangular mel filter banks.

	Inputs:
		M - number of filterbanks.
		NFFT - is the length of each filter (NFFT/2 + 1 typically).
		fs - sampling frequency.

	Outputs:
		H - triangular mel filterbank matrix.

	Reference: 
		Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing: 
		A guide to theory, algorithm, and system development. 
		Prentice Hall, Upper Saddle River, NJ, USA (pp. 315).
	'''
	fl = 0 # lowest frequency (Hz).
	fh = fs/2 # highest frequency (Hz).
	NFFT = int(NFFT) # ensure integer.
	H = np.zeros([M, NFFT], dtype=np.float32) # mel filter bank.
	for m in range(1, M + 1):
		bl = bpoint(m - 1, M, NFFT, fs, fl, fh) # lower boundary point, f(m - 1) for m-th filterbank.
		c = bpoint(m, M, NFFT, fs, fl, fh) # m-th filterbank centre point, f(m).
		bh = bpoint(m + 1, M, NFFT, fs, fl, fh) # higher boundary point f(m + 1) for m-th filterbank.
		for k in range(NFFT):
			if k >= bl and k <= c:
				H[m-1,k] = (k - bl)/(c - bl) # m-th filterbank up-slope. 
			if k >= c and k <= bh:
				H[m-1,k] = (bh - k)/(bh - c) # m-th filterbank down-slope. 
	return H

def melfbank_tapered(M, NFFT, fs):
	'''
	creates tapered triangular mel filter banks.

	Inputs:
		M - number of filterbanks.
		NFFT - is the length of each filter (NFFT/2 + 1 typically).
		fs - sampling frequency.

	Outputs:
		H - triangular mel filterbank matrix.

	Reference: 
		Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing: 
		A guide to theory, algorithm, and system development. 
		Prentice Hall, Upper Saddle River, NJ, USA (pp. 315).
	'''
	fl = 0 # lowest frequency (Hz).
	fh = fs/2 # highest frequency (Hz).
	NFFT = int(NFFT) # ensure integer.
	H = np.zeros([M, NFFT], dtype=np.float32) # mel filter bank.
	for m in range(1, M + 1):
		bl = bpoint(m - 1, M, NFFT, fs, fl, fh) # lower boundary point, f(m - 1) for m-th filterbank.
		c = bpoint(m, M, NFFT, fs, fl, fh) # m-th filterbank centre point, f(m).
		bh = bpoint(m + 1, M, NFFT, fs, fl, fh) # higher boundary point f(m + 1) for m-th filterbank.
		for k in range(NFFT):
			if k >= bl and k <= c:
				H[m-1,k] = (2*(k - bl))/((bh - bl)*(c - bl)) # m-th filterbank up-slope. 
			if k >= c and k <= bh:
				H[m-1,k] = (2*(bh - k))/((bh - bl)*(bh - c)) # m-th filterbank down-slope. 
	return H
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.python.ops.signal import window_ops
import functools
import numpy as np
import scipy.special as spsp
import tensorflow as tf

"""
[1] Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing:
	A guide to theory, algorithm, and system development.
	Prentice Hall, Upper Saddle River, NJ, USA (pp. 315).
"""

class AnalysisSynthesis:
	"""
	Analysis and synthesis stages of speech enhacnement.
	"""
	def __init__(self, N_d, N_s, K, f_s):
		"""
		Argument/s:
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
		"""
		self.N_d = N_d
		self.N_s = N_s
		self.K = K
		self.f_s = f_s
		self.W = functools.partial(window_ops.hamming_window,
			periodic=False)
		self.ten = tf.cast(10.0, tf.float32)
		self.one = tf.cast(1.0, tf.float32)

	def polar_analysis(self, x):
		"""
		Polar-form acoustic-domain analysis.

		Argument/s:
			x - waveform.

		Returns:
			Short-time magnitude and phase spectrums.
		"""
		STFT = tf.signal.stft(x, self.N_d, self.N_s, self.K,
			window_fn=self.W, pad_end=True)
		return tf.abs(STFT), tf.math.angle(STFT)

class SubbandFeatures(AnalysisSynthesis):
	"""
	Computes the input and target of Deep Xi.
	"""
	def __init__(self, N_d, N_s, K, f_s, M):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			M - number of filters.
		"""
		self.M = M
		self.H = self.mel_filterbank(self.M)

	def observation(self, x, x_len):
		"""
	    Observations for training (LSSEs).

		Argument/s:
			x - speech (dtype=tf.int32).
			x_len - speech length without padding (samples).

		Returns:
			x_STMS - short-time magnitude spectrum.
			x_STPS - short-time phase spectrum.
		"""
		x = self.normalise(x)
		x_LSSE = self.lsse(x)
		return tf.boolean_mask(x_LSSE, tf.sequence_mask(self.n_frames(x_len)))

	def lsse(self, x):
		"""
	    Compute log-spectral subband energies (LSSEs).

		Argument/s:
			x - noisy speech (dtype=tf.int32).
			x_len - noisy speech length without padding (samples).

		Returns:
			x_STMS - short-time magnitude spectrum.
			x_STPS - short-time phase spectrum.
		"""
		x_STMS, _ = self.polar_analysis(x)
		x_POW = tf.math.square(x_STMS)
		x_SSE = tf.linalg.matmul(x_POW, self.H, transpose_b=True)
		x_LSSE = tf.math.log(x_SSE)
		return x_LSSE

	def normalise(self, x):
		"""
		Convert waveform from int32 to float32 and normalise between [-1.0, 1.0].

		Argument/s:
			x - tf.int32 waveform.

		Returns:
			tf.float32 waveform between [-1.0, 1.0].
		"""
		return tf.truediv(tf.cast(x, tf.float32), 32768.0)

	def n_frames(self, N):
		"""
		Returns the number of frames for a given sequence length, and
		frame shift.

		Argument/s:
			N - sequence length (samples).

		Returns:
			Number of frames
		"""
		return tf.cast(tf.math.ceil(tf.truediv(tf.cast(N, tf.float32), tf.cast(self.N_s, tf.float32))), tf.int32)

	def mel_filterbank(self, M):
		"""
		Created a mel-scale filterbank using the equations from [1].
		The notation from [1] is also used. For this case, each filter
		sums to unity, so that it can be used to weight the STMS a
		priori SNR to compute the a priori SNR for each subband, i.e.
		each filter bank.

		Argument/s:
			M - number of filters.

		Returns:
			H - triangular mel filterbank matrix.

		"""
		f_l = 0 # lowest frequency (Hz).
		f_h = self.f_s/2 # highest frequency (Hz).
		K = self.K//2 + 1 # number of frequency bins.
		H = np.zeros([M, K], dtype=np.float32) # mel filter bank.
		for m in range(1, M + 1):
			bl = self.bpoint(m - 1, M, f_l, f_h) # lower boundary point, f(m - 1) for m-th filterbank.
			c = self.bpoint(m, M, f_l, f_h) # m-th filterbank centre point, f(m).
			bh = self.bpoint(m + 1, M, f_l, f_h) # higher boundary point f(m + 1) for m-th filterbank.
			for k in range(K):
				if k >= bl and k <= c:
					H[m-1,k] = (2*(k - bl))/((bh - bl)*(c - bl)) # m-th filterbank up-slope.
				if k >= c and k <= bh:
					H[m-1,k] = (2*(bh - k))/((bh - bl)*(bh - c)) # m-th filterbank down-slope.
		return H

	def bpoint(self, m, M, f_l, f_h):
		"""
		Detirmines the frequency bin boundary point for a filterbank.

		Argument/s:
			m - filterbank.
			M - total filterbanks.
			f_l - lowest frequency.
			f_h - highest frequency.

		Returns:
			Frequency bin boundary point.
		"""
		K = self.K//2 + 1 # number of frequency bins.
		return ((2*K)/self.f_s)*self.mel_to_hz(self.hz_to_mel(f_l) + \
			m*((self.hz_to_mel(f_h) - self.hz_to_mel(f_l))/(M + 1))) # boundary point.
	def log_10(self, x):
		"""
		log_10(x).

		Argument/s:
			x - input.

		Returns:
			log_10(x)
		"""
		return tf.truediv(tf.math.log(x), tf.math.log(self.ten))
	def hz_to_mel(self, f):
		"""
		Converts a value from the Hz scale to a value in the mel scale.

		Argument/s:
			f - Hertz value.

		Returns:
			Mel value.
		"""
		return 2595*np.log10(1 + (f/700))

	def mel_to_hz(self, m):
		"""
		Converts a value from the mel scale to a value in the Hz scale.

		Argument/s:
			m - mel value.

		Returns:
			Hertz value.
		"""
		return 700*((10**(m/2595)) - 1)

	def log_10(self, x):
		"""
		log_10(x).

		Argument/s:
			x - input.

		Returns:
			log_10(x)
		"""
		return tf.truediv(tf.math.log(x), tf.math.log(self.ten))

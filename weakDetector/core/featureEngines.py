import torchaudio
import torch
import math
import scipy.io
from scipy.signal import butter, lfilter
import scipy
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as T

from weakDetector.utils.func import renormalise, bandpass, moving_average

#from abc import ABC, abstractmethod


# TODO TEST THESE CCREATEORS

class FeatureEngine():

	def __init__(self, window_size, target_length=None, sampling_rate=48000, bp_freqs=None, bp_order=6):
		"""Initialize FeatureEngine abstract class.

		Args:
			window_size (int): Length of non-overlapping windows to perform feature extraction on
			target_length (int, optional): Expected length of wavfile (in samples) after resampling. Defaults to None.
			sampling_rate (int, optional): Target sampling rate after preprocessing. Defaults to 48000.
			bp_freqs (list, optional): Low and high frequency limits of bandpass filter to apply during preprocessing. Defaults to None.
			bp_order (int, optional): Order of bandpass filter to apply if applicable. Defaults to 6.
		"""
		self._window_size=window_size
		self._target_length = target_length
		self._target_sr = sampling_rate
		self._bp_freqs = bp_freqs
		self._bp_order = bp_order

		self._tf = torchaudio.transforms.Spectrogram(
			n_fft = 512,
			window_fn = torch.hann_window,
			hop_length = int(256)
		)

	@property 	
	def target_length(self):
		""" Get target_length. """
		return self._target_length

	@target_length.setter
	def target_length(self, value):
		self._target_length = value


	def preprocess_wavfile(self, w, sr):
		"""
		TODO
		"""
		# substract median
		w = w - torch.median(w)
		
		# resample if specified and necessary
		w  = self._resample_if_necessary(w, sr)

		#keep first channel
		if w.shape[0]>1:
			w =  w[:1, :]

		# cut to specified length if signal is too long
		if self._target_length is not None:
			w = self._cut_if_necessary(w)

		# Filter if specified
		w = self._filter_if_specified(w)


		#Standardise file
		w = w - torch.median(w)
		w = w/torch.std(w)


		return w

	def load_file(self, fpath):
		"""Load file from a specified fpath and do some basic acoustic preprocessing.

		Args:
			fpath (string): Absolute path of the wavfile to load

		Returns:
			torch.Tensor: Processed (resampled, filtered, cut to a specified length)  
						  timeseries
		"""
		
		# load wavfile
		w, sr = torchaudio.load(fpath)

		#preprocess
		w = self.preprocess_wavfile(w, sr)
		


		return w
	



	def _resample_if_necessary(self, w, sr):
		"""Resample time series if sampling rate is different from target_sr property.

		Args:
			w (torch.Tensor): Time series
			sr (int): Sampling rate of w

		Returns:
			torch.Tensor : Time series at sampling rate of target_sr
		"""
		if self._target_sr!=sr:
			resampler = torchaudio.transforms.Resample(sr, self._target_sr)
			w = resampler(w)
		return w

	def _cut_if_necessary(self, w, dim=1):
		"""Cut signal to target_length property along dimension dim.

		Args:
			w (torch tensor): waveform signal to cut.
			dim (int, optional): Dimension to cut along. Defaults to 1.

		Raises:
			ValueError: Specified dimension does not exist in signal

		Returns:
			torch tensor: Cut signal, or original signal if cutting was not needed. 
		"""

		if dim>=len(w.shape):
			raise ValueError('Specified dimension is larger than number of dimensions of signal.')

		if w.shape[dim] > self._target_length:		
			slicing = [slice(None)] * w.ndimension()
			slicing[dim] = slice(0, self._target_length)
			w = w[tuple(slicing)]

		return w

	def _filter_if_specified(self, w):
		"""Apply bandpass filter if specified in property bp_freqs.

		Args:
			w (torch.Tensor): time series

		Returns:
			torch.Tensor: time series after bandpass filter (if applicable) or original timeseries
		"""
		if self._bp_freqs is not None:
			lowcut, highcut =  self._bp_freqs
			w = bandpass(w, fs=self._target_sr, lowcut=lowcut, highcut=highcut, order=self._bp_order , )
		return w

	def __call__(self, fpath):
		"""Load file and extract feature sequence.

		Args:
			fpath (str): Absolute path of file to process.

		Returns:
			torch.Tensor: Feature sequence
		"""
		#Load file
		w = self.load_file(fpath)
		#Extract features
		seq = self.extract(w)
		
		return seq	

	def extract(self, w):
		"""Extract feature sequence from signal.

		Args:
			w (_type_): _description_

		Returns:
			_type_: _description_
		"""
		pass

class HeuristicFeatureExtractor(FeatureEngine):
	# TODO maybe improve this class so that order of fequencies can't be fucked up bymessing with the code... eg store descriptions and functions together in tuple
	def __init__(self, window_size, target_length=None, sampling_rate=48000, bp_freqs=None, bp_order=6,
			  rms = False, rms_freqs = [1000, 20000],   mean_freq=False, peak_freq=False, 
			  energy_sums=False,energy_bands = [1000, 4000, 8000, 16000],  spectral_width=False):
		"""Initialize HeuristicFeatureExtractor class.

		Args:
			window_size (int): Length of non-overlapping windows to perform feature extraction on
			target_length (int, optional): Expected length of wavfile (in samples) after resampling. Defaults to None.
			sampling_rate (int, optional): Target sampling rate after preprocessing. Defaults to 48000.
			bp_freqs (list, optional): Low and high frequency limits of bandpass filter to apply during preprocessing. Defaults to None.
			bp_order (int, optional): Order of bandpass filter to apply if applicable. Defaults to 6.
			rms (bool, optional): Whether to include rms computations of windows in feature vector. Defaults to False.
			rms_freqs (list, optional): Limits of frequency bands over which to compute rms values. Defaults to [1000, 20000].
			mean_freq (bool, optional): Whether to include mean frequency of window in feature vector. Defaults to False.
			peak_freq (bool, optional): Whether to include peak frequency of window in feature vector. Defaults to False.
			energy_sums (bool, optional): Whether to include energy sums of window over specified frequency bands in feature vector. Defaults to False.
			energy_bands (list, optional): Limits of frequency bands over which to computer energy sums. Defaults to [1000, 4000, 8000, 16000].
			spectral_width (bool, optional): Whether to include spectral width of window in feature vector. Defaults to False.
		"""
		
		# Parent init
		super().__init__(window_size, target_length, sampling_rate, bp_freqs, bp_order)

		self._rms = rms
		self._peak_freq = peak_freq
		self._mean_freq = mean_freq

		self._energy_sums = energy_sums
		self._spectral_width = spectral_width

		if  rms:
			self._rms_freqs = rms_freqs
		if energy_sums:
			self._energy_bands = energy_bands

		self._latent_size = self._compute_latent_size()
		self._feature_descriptions = self._create_feature_description()
	@property
	def latent_size(self):
		"""Get latent size."""
		return self._latent_size

	@property
	def feature_descriptions(self):
		"""Get feature descriptions"""
		return self._feature_descriptions


	def extract(self, w):
		# Start empty sequence of features
		len_seq = int(w.shape[1]/self._window_size)
		seq = torch.empty(self.latent_size, len_seq)

		#Create windows and extract features
		for i in range(len_seq):
			#get chunk
			w_i = w[0, i*self._window_size:(i+1)*self._window_size]
			seq[:, i] = self._extract_features_from_window(w_i)
		return seq



	def _compute_latent_size(self):
		"""Compute latent size based on included features specified at init.

		Returns:
			int: Number of features
		"""
		latent_size=0
		if self._rms:
			latent_size += len(self._rms_freqs)-1
		if self._peak_freq:
			latent_size+=1
		if self._mean_freq:
			latent_size+=1
		if self._energy_sums:
			latent_size+=len(self._energy_bands)-1
		if self._spectral_width:
			latent_size+=1
		return latent_size
	
	def _create_feature_description(self):
		"""Create description of extracted features

		Returns:
			list: descriptions of extracted features
		"""
		descriptions = []
		if self._rms:
			for i in range(len(self._rms_freqs)-1):
				descriptions.append(f'RMS {self._rms_freqs[i]}Hz-{self._rms_freqs[i+1]}Hz')
		if self._peak_freq:
			descriptions.append('Peak frequency')
		if self._mean_freq:
			descriptions.append('Mean frequency')
		if self._energy_sums:
			for i in range(len(self._energy_bands)-1):
				descriptions.append(f'Energy sum {self._energy_bands[i]}Hz-{self._energy_bands[i+1]}Hz')
		if self._spectral_width:
			descriptions.append('Spectral width')
		return descriptions
		
	def _extract_features_from_window(self, w_i):
		"""Extract features from a window.

		Args:
			w_i (torch.Tensor): Time-series window

		Returns:
			torch.Tensor: Extracted frequencies.
		"""
		features = []
		if self._rms:
			features  += self._compute_rms_values(w_i) 
			#print('rms')
		
		if len(features)<self._latent_size:
			# We need some freq and energy computatuins
			s, freqs = self._compute_sfft_s(w_i, self._target_sr)

		if self._peak_freq:
			features += [self._compute_peak_freq(s, freqs)]
			#print('peak freqs')
		if self._mean_freq:
			#print('mean freqs')
			features += [self._compute_mean_freq(s, freqs)]
		
		if self._energy_sums:
			#print('features:', features)
			#print('energy sums:', self._compute_energy_sums(s,freqs))
			features+= self._compute_energy_sums(s, freqs)

		if self._spectral_width:
			features += [self._compute_spectral_width(w_i)]

		return torch.tensor(features, dtype=torch.float32)

	def _compute_rms_values(self, w):
		"""Compute all rms values in one window over specified frequency bands.

		Args:
			w (torch.Tensor): time series

		Returns:
			list: rms values
		"""
		rms_values = []
		for i in range(len(self._rms_freqs)-1):
			low_freq = self._rms_freqs[i]
			high_freq = self._rms_freqs[i+1]
			b,a = butter(self._bp_order, [low_freq, high_freq], fs=self._target_sr, btype='band')
			w_f = torch.tensor(lfilter(b, a, w), dtype=torch.float32)

			# Compute rms
			rms_values.append(self._root_mean_square(w_f))
		
		return rms_values

	def _root_mean_square(self, w):
		"""Compute RMS of time series.

		Args:
			w (torch.Tensor): time series

		Returns:
			Float: RMS 
		"""
		if len(w.shape)>1:
			w = w.flatten()
		scalar_prod = torch.dot(w, w)
		rms = math.sqrt(scalar_prod/w.shape[0])
		return rms

	def _compute_sfft_s(self, w, nfft=512):
		"""Compute fft associated values.

		Args:
			w (torch tensor): waveform
			sr (int): sample rate
			nfft (int, optional): fft size. Defaults to 512.

		Returns:
			np.array: freq vector
			np.array: squared smoothed spectral vector s
		"""
		w = w.numpy()
		sfft = scipy.fft.fft(w, axis=0, n=nfft)[:nfft//2]
		freq = scipy.fft.fftfreq(nfft, d=1/self._target_sr)[:nfft//2]
		s = np.array([np.abs(sfft[i]*sfft[i]) for i in range(nfft//2)])

		smooth_s = moving_average(s)


		return freq, smooth_s#, smooth_db

	def _compute_peak_freq(self, s, freqs):
		"""Compute peak frequency from spectral vector.

		Args:
			s (numpy.array): Squared spectral vector (smoothed)
			freqs (numpy.array): Vector of frequencies linked to s.

		Returns:
			Float: peak frequency
		"""
		ind_max_freq = np.where(s==s.max())[0][0]
		return freqs[ind_max_freq]
	
	def _compute_mean_freq(self, s, freqs):
		"""Compute mean frequency from spectral vector

		Args:
			s (numpy.array): Squared spectral vector (smoothed)
			freqs (numpy.array): Vector of frequencies linked to s.

		Returns:
			Float: mean frequency
		"""
		return np.dot(s, freqs)/np.sum(s)

	def _compute_energy_sums(self, s, freqs):
		"""Generate energy sums over specified frequency bands.

		Args:
			s (numpy.array): Squared spectral vector (smoothed)
			freqs (numpy.array): Vector of frequencies linked to s.

		Returns:
			list: energy sums.
		"""

		lowcuts =  self._energy_bands[:-1]
		highcuts = self._energy_bands[1:]

		ens = []

		for i in range(len(lowcuts)):
			lc = lowcuts[i]
			hc = highcuts[i]
			energy = np.sum(s[np.logical_and(freqs>=lc, freqs<hc)])#/(hc-lc)
			ens.append(energy)
			
		return ens

	def _compute_spectral_width(self, w, nfft=256, db_diff=8, max_sep=3):
		"""Get spectral width of wavform db_diff from peak.

		Args:
			w (torch.Tensor): waveform
			nfft (int, optional): Fft size. Defaults to 256.
			db_diff (int, optional): Max difference from peak. Defaults to 8.
			max_sep (int, optional): Maximum separation of samples below db_diff to still be considered part of the peak. Defaults to 3.

		Returns:
			float: Spectral width.
		"""

		s, freqs = self._compute_sfft_s(w, nfft=nfft)
		#Compute s in db scale
		db =  10*np.log10(s)
		max_db = db.max()
		thresh = max_db-db_diff

		ind = [i for i in range(len(db)) if db[i]>=thresh]
		res = []
		tmp = []
		prv = ind[0]
		for l in ind:
			if l-prv > max_sep:
				res.append(tmp)
				tmp = []
			tmp.append(l)
			prv = l
		res.append(tmp)

		max_ind = np.where(db==max_db)[0][0]
		
		for l in res:
			if max_ind in l:
				width = freqs[l[-1]]-freqs[l[0]]
				break
			
		return width

class VAEFeatureExtractor(FeatureEngine):

	def __init__(self, window_size, latent_size, input_type, model, target_length=None, sampling_rate=48000, bp_freqs=None, bp_order=6,
			  device='cuda', n_parallel=1):

		# Parent init
		super().__init__(window_size, target_length, sampling_rate, bp_freqs, bp_order) 
		
		self._latent_size = latent_size
		self._input_type = input_type

		self._device = device
		self._model = model.to(self._device)

		self._prepare_input = self._choose_data_preparer()

		if n_parallel<=0:
			raise ValueError('Invalid value for n_parallel: {n_parallel}. Must be positive integer')
		if self._input_type=='spectrogram' and n_parallel!=1:
			n_parallel = 1
			warnings.warn("Parallel implementation not supported for spectrogram data type. Defaulting to n_parallel=1. ")
			
		self._n_parallel = n_parallel
		
		self._step = self._window_size * self._n_parallel



		if self._input_type=='spectrogram':
			self._tf_spec = T.Resize((128, 128))



	@property
	def latent_size(self):
		"""Return latent_size"""
		return self._latent_size
	
	@property
	def model(self):
		"""Get VAE model."""
		return self._model
	
	def _choose_data_preparer(self):
		"""Choose function to prepare data to be sent into encoer based on input_type property 

		Raises:
			ValueError: If input type property is not supported

		Returns:
			function: Preparer function
		"""
		if self._input_type == 'short_wf' or self._input_type=='long_wf':
			return self._prepare_waveform
		elif self._input_type == 'spectral_profile':
			return self._prepare_specprof
		elif self._input_type == 'spectrogram':
			return self._prepare_spectrogram
		else:
			raise ValueError(f'Invalid value for input_type: {self._input_type}. Must be one of "short_wf", "long_wf" , "spectral_profile", "spectrogram"')
	
	def _prepare_waveform(self, w_i):
		"""Prepare waveform to be sent into encoder.

		Args:
			w_i (torch.Tensor): Time series

		Returns:
			torch.Tensor: Reformatted Time series
		"""
		#print('Wi shape', w_i.shape)
		w_i = w_i.reshape(self._n_parallel, self._window_size)

		for k in range(self._n_parallel):
			w_i[k, :] = renormalise(w_i[k, :])

		r = w_i.reshape(self._n_parallel, 1, self._window_size)
		
		return r

	def _prepare_specprof(self, w_i):
		"""Create spectral profile to be sent into encoder

		Args:
			w_i (torch.Tensor): Time series

		Returns:
			torch.Tensor: Spectral profile
		"""
		w_i = w_i.reshape(self._n_parallel, self._window_size)
		#print('Wi shape', w_i.shape)
		norm_specs = torch.empty((self._n_parallel, 200))
		#print('norm_specs shape', norm_specs.shape)
		for k in range(self._n_parallel):
			spec = self._tf(w_i[k, :])
			clipped_spec = spec[10:210, 1]
			flipped_spec = torch.flipud(clipped_spec)

			smooth_spec = moving_average(flipped_spec, 8)
			normalised_spec = renormalise(smooth_spec)
			norm_specs[k, :] = torch.Tensor(normalised_spec)
		

		r = norm_specs.reshape(self._n_parallel, 1, 200)

		return r

	def _prepare_spectrogram(self, w_i):
		"""Create spectrogram to be sent into encoder.

		Args:
			w_i (torch.Tensor): Time series

		Returns:
			torch.Tensor: Spectrogram representation
		"""
		# Create spectrogram representation
		spec = self._tf(w_i)
		# Remove frequencies below 1kHz and above 20kHz, and reshape
		if len(spec.shape)==2:
			spec = spec.unsqueeze(0)

		resized_spec = self._tf_spec(spec[:, 10:210, :])
		# Flip spectrogram (not necessary but makes looking at them easier)
		flipped_spec = torch.flipud(resized_spec[0, :, :])

		#Standardise
	#	flipped_spec = (flipped_spec-flipped_spec.median())/flipped_spec.std()


		# Reshape and return
		return renormalise(flipped_spec.resize(1, 1, 128, 128))

	def _encode(self, w_i):
		"""Encode timeseries using the class' data input and model.

		Args:
			w_i (torch.Tensor): Time series

		Returns:
			torch.Tensor: Encoding
		"""
		r = self._prepare_input(w_i)
		with torch.no_grad():
			_,  means, logvars = self._model.encoder(r.float().to(self._device))
		embeddings = torch.concat((means.cpu(), logvars.cpu()), 1)
		embeddings = torch.transpose(embeddings, 0, 1).cpu()
		#torch.cuda.empty_cache()

		return embeddings



	def extract(self, w):
		"""Extract sequence of VAE embeddings

		Args:
			w (torch.Tensor): Loaded and processed signal.

		Returns:
			torch.Tensor: Embeddings sequence
		"""

		# Start empty sequence of features
		len_seq = int(w.shape[1]/self._window_size)
		
		seq = torch.empty(2*self.latent_size, len_seq)
		
		#Create windows and encode

		n_steps = int(w.shape[1]/self._step)

		for i in range(n_steps):
			#get chunk
			w_i = w[0, i*self._step:(i+1)*self._step]
			#print(w.shape[1], i*self._step, (i+1)*self._step)
			seq[:, i:(i+self._n_parallel)] = self._encode(w_i)

		return seq		



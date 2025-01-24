import numpy as np
import torch
from scipy.signal import butter, lfilter


def renormalise (signal, tmin=0, tmax=1):
	"""Normalise signal.

	Args:
		signal (torch.Tensor): Signal to normalise
		tmin (int, optional): Minimum value of normalised signal. Defaults to 0.
		tmax (int, optional): Maximum value of normalised signal. Defaults to 1.

	Returns:
		torch.Tensor: Normalised signal
	"""
	min_val = signal.min()
	max_val = signal.max()

	normalised_signal = ((signal-min_val)/(max_val-min_val))*(tmax-tmin)+tmin
	return normalised_signal


def standardise(signal):
	signal = (signal - signal.median())/signal.std()
	
	return signal

def identity(signal):
	return signal

def bandpass(signal, fs, lowcut, highcut, order=6):
	# Auxiliary function to apply butterworth bandpass filter to waveform.
	# Input: waveform, sampling rate, lower frequency, higher frequency, order of filter (default = 5)
	b,a = butter(order, [lowcut, highcut], fs=fs, btype='band')
	filtered_signal = torch.tensor(lfilter(b, a, signal))
	return filtered_signal


def moving_average(a, n=5):
	"""Smoothes sequence using moving average

	Args:
		a (_type_): _description_
		n (int, optional): _description_. Defaults to 5.

	Returns:
		_type_: _description_
	"""
	inf_ind = list(np.where(np.isneginf(a))[0])
	if len(inf_ind)==len(a):
		return -1

	for i in inf_ind:
		if i==0:
			a[0]=a[1]
		if (i>0) & (i<(len(a)-1)):
			if not (i+1) in inf_ind:
				a[i] = (a[i-1]+a[i+1])/2
			else:
				a[i]=a[i-1]
		if i ==(len(a)-1):
			a[i] = a [i-1]
	cs = np.cumsum(a, dtype=float)
	ma = np.copy(cs)
	half_n = n//2
	for i in range(len(ma)):
		min_ind = max(0, i-half_n)
		max_ind = min(len(ma)-1, i+half_n+1) 
		ma[i] = (cs[max_ind]-cs[min_ind])/(max_ind-min_ind)
		
	return ma
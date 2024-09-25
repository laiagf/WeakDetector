import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from weakDetector.utils.func import moving_average, renormalise
from config import ROOT_DIR

class ClickDataset(Dataset):
	def __init__(self, dataset_features, csv_file, tensor_dir, normalise=True, smooth=False, channels='all', sources='all', split_files=None):
		"""Initialize ClickDataset dataset.

		Args:
			dataset_features (_type_): _description_
			csv_file (_type_): _description_
			tensor_dir (_type_): _description_
			normalise (bool, optional): Whether to normalize the tensors. Defaults to True.
			smooth (bool, optional): Whether to apply smoothing to the tensors. Defaults to False.
			channels (str or int or list, optional): Channels to include in the tensors. Defaults to 'all'.
			sources (str or list, optional): Sources from which to include tensors. Defaults to 'all'.
			split_files (_type_, optional): _description_. Defaults to None.

		Raises:
			ValueError: _description_
		"""

		# Read CSV file into a DataFrame	   
		

		self._dataset_features = dataset_features
		self._feature_length, self._tensor_shape = self._calculate_dataset_parameters()
		df = pd.read_csv(os.path.join(ROOT_DIR, csv_file))

		if split_files!=None and sources!='all':
			raise ValueError(f"Filtering by both index and source is not allowed")

		if split_files:
			df = df[df.File.isin(split_files)]
			df.reset_index(drop=True, inplace=True)

		# Filter DataFrame based on specified sources
		if sources!='all':
			df = df[df.Dataset.isin(sources)]
			df.reset_index(drop=True, inplace=True)
		# Store annotations and parameters			
		self._annotations = df
		self._tensor_dir = tensor_dir
		self._smooth = smooth
		self._channels = channels
		self._normalise = normalise

	@property
	def feature_length(self):
		"""
		Get length of features.
		"""
		return self._feature_length
    
	def __len__(self):
		"""
		Return the number of samples in the dataset.
		"""
		return len(self._annotations)
	
	def __getitem__(self, index):
		"""
		Retrieve a sample from the dataset at the given index.

		Args:
			index (int): Index of the sample to retrieve.

		Returns:
			torch.Tensor: Processed tensor data.
		"""
		# Load tensor data from file		
		tensor_path = os.path.join(self._tensor_dir, self._annotations.Tensor_name[index])
		tensor = torch.load(tensor_path)
		# Replace NaN values with zeros		
		tensor = torch.nan_to_num(tensor)

		# Apply filters and normalisation if specified
		if self._smooth:
			tensor = moving_average(tensor, 8)
		if self._channels!='all':
			tensor = tensor[self._channels, :]	
		if self._normalise:
			tensor = renormalise(tensor)
		
		# Reshape to specified shape 
		tensor = tensor.reshape(self._tensor_shape)

		return tensor.float()

	def _calculate_dataset_parameters(self):
		"""Get expected tensor shape and window length 

		Raises:
			ValueError: _description_

		Returns:
			_type_: _description_
		"""
		if self._dataset_features == 'long_wf':
			tensor_shape = [1, 2048]
			feature_length = 2048

		elif self._dataset_features == 'short_wf':
			tensor_shape = [1, 512]
			feature_length = 512

		elif self._dataset_features == 'spectral_profile':
			feature_length = 200
			tensor_shape = [1, 200]

		elif self._dataset_features == 'spectrogram' or self._dataset_features=='std_spectrogram':
			feature_length = -1
			tensor_shape = [1, 128, 128]

		else:
			raise ValueError(f"Unsupported dataset: {self._dataset_features}")

		return feature_length, tensor_shape




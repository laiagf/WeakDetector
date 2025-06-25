import pandas as pd
pd.options.mode.chained_assignment = None
import torch
import os
from torch.utils.data import Dataset
from weakDetector.utils.func import standardise
from weakDetector.config import ROOT_DIR

class SpermWhaleDataset(Dataset):
	def __init__(self, annotations_file, files_dir, df_standard=None, target_length=None, sources='all', channels='all', min_snr=0):
		"""Initialise SpermWhaleDataset dataset.

		Args:
			annotations_file (_type_): _description_
			files_dir (_type_): _description_
			target_length (_type_): _description_
			sources (str, optional): _description_. Defaults to 'all'.
			channels (str, optional): _description_. Defaults to 'all'.
		"""

		if isinstance(annotations_file, pd.DataFrame):
			self._df_annotations = annotations_file
			if min_snr!=0 and 'SNR_999' not in self._df_annotations.columns:
				raise ValueError(f"Min SNR specified ({min_snr}) but SNR_999 not in annotations file columns")
			if min_snr>0 and min(self._df_annotations.SNR_999)<min_snr:
				raise ValueError(f"Annotations dataframe must have SNR_999 >= {min_snr}.")
			if sources!='all' and not all(s in self._df_annotations.Dataset.unique() for s in sources):
				raise ValueError(f"Annotations dataframe must have Dataset in {sources}.")
		else:
			self._df_annotations = self._load_annotations(annotations_file, sources, min_snr)

		self._files_dir = files_dir

		self._target_length = target_length

		self._channels = channels


		if isinstance(df_standard, pd.DataFrame):

			standard_dict = {
				(row['Dataset'], row['dimension']): {'mean': row['row_mean'], 'std': row['row_std']}
				for _, row in df_standard.iterrows() }
			self._standard_dict =  standard_dict
			self._standardise=True
		else:
			self._standardise=False


		print(f"There are {len(self._df_annotations)} in the SpermWhaleDataset") 


	def _load_annotations(self, annotations_file, sources, min_snr):
		"""Load dataframe of files and labels

		Args:
			annotations_file (_type_): _description_
			sources (_type_): _description_

		Returns:
			_type_: _description_
		"""
		df_annotations = pd.read_csv(os.path.join(ROOT_DIR, annotations_file))
		
		if min_snr>0:
			df_annotations = df_annotations[df_annotations.SNR_999>=min_snr].reset_index(drop=True)

		if sources!='all':
			df_annotations = df_annotations[df_annotations.Dataset.isin(sources)].reset_index(drop=True)

		return df_annotations

	@property
	def annotations(self):
		"""Get annotations dataframe.

		Returns:
			_type_: _description_
		"""
		return self._df_annotations
	
	@property
	def files_dir(self):
		return self._files_dir
	

	@annotations.setter
	def annotations(self, df):
		self._df_annotations = df.reset_index(drop=True)
	def __len__(self):
		return len(self._df_annotations)
	
	def load_item(self, fname):
		"""Load sequence of extracted features.

		Args:
			fname (_type_): _description_

		Returns:
			_type_: _description_
		"""
		try:
			t = torch.load(os.path.join(self._files_dir, fname))
		except Exception:
			print(os.path.join(self._files_dir, fname))
			raise Exception(f"Error loading file {fname} in {self._files_dir}")
		t = torch.nan_to_num(t)

		
		dataset = '_'.join(fname.split('_')[:-2])

		# cut and get selected columns
		if self._channels!='all':
			t = t[self._channels, :]
		if self._target_length:
			t = t[:, :self._target_length]

		
		t = self._row_standardise(t, dataset)
		if self._target_length:
			t = self._right_pad_if_necessary(t)


		t = torch.nan_to_num(t)

		return t
	
	def _right_pad_if_necessary(self, signal, dim=1):
		"""Right pad signal if below specified length.

		Args:
			signal (_type_): _description_
			dim (int, optional): _description_. Defaults to 1.

		Returns:
			_type_: _description_
		"""
		length_signal = signal.shape[dim]
		if length_signal < self._target_length:
			num_missing_samples = self._target_length - length_signal
			last_dim_padding = (0, num_missing_samples) # padding if necessaryy
			signal = torch.nn.functional.pad(signal, last_dim_padding)
		return signal  
	


	def _row_standardise(self, t, dataset):

		if self._standardise:
			t = torch.clamp(t, min=-1000, max=1000)

			for i in range(t.shape[0]):
				m, std = self._standard_dict[dataset, i]['mean'], self._standard_dict[dataset, i]['std']
				if std!=0:
					t[i, :] = (t[i, :]- m)/std
				else:
					t[i, :] = (t[i, :] -m)
		return t

	def __getitem__(self, index):
		label = self._df_annotations.Label[index]
		sequence = self.load_item(self._df_annotations.FileName[index][:-3]+'pt')
		return sequence, int(label)
	

	



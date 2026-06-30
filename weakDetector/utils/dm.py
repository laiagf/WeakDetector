# Dataset management functions
import pandas as pd
pd.options.mode.chained_assignment = None
from weakDetector.datasets.spermWhaleDataset import SpermWhaleDataset
from weakDetector.datasets.clickDataset import ClickDataset
from weakDetector.utils.mm import get_target_length

import os
import torch
import copy
from weakDetector.config import ROOT_DIR, SOURCES


def random_split_df(csv_path, proportions = [0.7, 0.15, 0.15], manual_seed=0):
	"""Get filenames after random split of df.

	Args:
		csv_path (string): path to 
		proportions (list, optional): Proportions of training, validation and test sets when randomly splitting. Defaults to [0.7, 0.15, 0.15].
		manual_seed (int, optional): _description_. Defaults to 0.

	Returns:
		_type_: _description_
	"""
	df = pd.read_csv(os.path.join(ROOT_DIR, csv_path)) # TODO how do we handle root dir here
	total_size = len(df)
	# Calculate sizes of the train and validation sets
	train_size = int(proportions[0] * total_size)
	val_size = int(proportions[1] * total_size)

	# Generate the same random indices as before
	indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(manual_seed))

	# Use the same indices to split into train and validation sets
	train_indices = indices[:train_size].tolist()
	train_files = list(set(df.File[train_indices]))

	val_indices = indices[train_size:train_size+val_size].tolist()
	val_files = list(set(df.File[val_indices]))

	test_indices = indices[train_size+val_size:].tolist()
	test_files = list(set(df.File[test_indices]))
	return train_files, val_files, test_files


# TODO rethink this this needs to be in the class itself or something like that
def split_sw_dataset(dataset:SpermWhaleDataset, cfg, split_proportions = [0.7, 0.15, 0.15]):
	"""Split SpermWhaleDataset into training and validation sets.

	Args:
		dataset (SpermWhaleDataset): Dataset to split.
		cfg (omegaconf.dictconfig.DictConfig): Config dictionary.
		proportions (list, optional): Proportions of training, validation and test sets when randomly splitting. Defaults to [0.7, 0.15, 0.15].

	Raises:
		ValueError: _description_

	Returns:
		tuple: _description_
	"""
	if 'split' not in cfg.keys() or cfg['split']=='random':
		split_ns = [int(len(dataset)*p) for p in split_proportions]
		split_ns[-1] = len(dataset) - sum(split_ns[:-1])  # Adjust the last split to account for rounding errors

		train_set, val_set, test_set = torch.utils.data.random_split(dataset, split_ns, generator=torch.Generator().manual_seed(cfg.random_state)) 
		df_dataset = dataset.annotations
		df_dataset['split']=''
		df_dataset.split[train_set.indices]='train'
		df_dataset.split[val_set.indices]='val'
		df_dataset.split[test_set.indices]='test'



	elif cfg['split']=='by_source':
		# For transferability experiments. train_set and val_set coming from train_sources, test_Set coming from the rest
		#train_set = dataset
		test_set_sources = [s for s in SOURCES if s not in cfg['train_sources']]
		print('Test sources', test_set_sources)

		#val_set = SpermWhaleDataset(annotations_file=cfg.annotations_file,
		#						files_dir=dataset.files_dir,
		#						target_length=get_target_length(cfg),
		#						sources=val_set_sources, min_snr=cfg.min_snr,
		#						channels=dataset._channels) 
		
		#Copy train set into test set and just change annotations file
		test_set = copy.deepcopy(train_set)
		test_set.annotations = test_set._load_annotations(cfg.annotations_file, test_set_sources, cfg.min_snr)

		train_set_split = split_proportions[0]*len(dataset)
		val_set_split = len(dataset)-train_set_split 
		train_set, val_set = torch.utils.data.random_split(dataset, [train_set_split, val_set_split], generator=torch.Generator().manual_seed(cfg.random_state)) 

		df_train_val = dataset.annotations	
		df_train_val['split'] = 0
		df_train_val.split[train_set.indices]='train'
		df_train_val.split[val_set.indices]='val'		



		def_test = test_set.annotations
		df_test['split'] = 'test'
		df_dataset = pd.concat([df_train_val, df_test]).reset_index(drop=True)
	
	else:
		raise ValueError(f"Unsuported dataset split method {cfg['split']}.")
	
	return train_set, val_set, test_set, df_dataset


def split_click_dataset(dataset:ClickDataset, cfg, proportions=[0.7, 0.15, 0.15]):
	"""Split ClickDataset into training and validation sets.

	Args:
		dataset (ClickDataset): Dataset to split.
		cfg (omegaconf.dictconfig.DictConfig): Config dictionary.
		proportions (list, optional): Proportions of training, validation and test sets when randomly splitting. Defaults to [0.7, 0.15, 0.15].

	Raises:
		ValueError: _description_

	Returns:
		tuple: _description_
	"""
	if 'split' not in cfg.keys() or cfg['split']=='random' or cfg['train_sources']=='all':
		split_ns = [int(len(dataset)*p) for p in split_proportions]
		split_ns[-1] = len(dataset) - sum(split_ns[:-1])  # Adjust the last split to account for rounding errors
		train_set, val_set, test_set = torch.utils.data.random_split(dataset, split_ns,  generator=torch.Generator().manual_seed(cfg.random_state)) 
		print(f"There are {len(dataset)} samples in the dataset, random split ({len(train_set)} training / {len(val_set)} validation) / {len(test_set)} test.")
	
	elif cfg.split=='by_source':
		#split dataset into train and val, and then test_set comes from sources not in the dataset already

		train_set_split = split_proportions[0]*len(dataset)
		val_set_split = len(dataset)-train_set_split 
		train_set, val_set = torch.utils.data.random_split(dataset, [train_set_split, val_set_split], generator=torch.Generator().manual_seed(cfg.random_state)) 		

		test_sources = [s for s in SOURCES if s not in cfg.train_sources]
		test_set = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=test_sources)
		print(f"There are {len(train_set)} samples in the training dataset / {len(val_set)} in the val_set (sources{cfg.train_sources}, and {len(test_set)} in the test set (sources {test_sources})")
	
	elif cfg.split=='by_tcn_4':
		train_files, val_files, test_files = random_split_df('files/datasets/4mindataset.csv', proportions=proportions, manual_seed=cfg.random_state)
		train_set =  ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=train_files)
		val_set = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=val_files)
		test_set = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=test_files)
		print(f"There are {len(train_set)} samples in the training dataset, {len(val_set)} in the val set and {len(test_set)} in the test set. Split made according to random split on 4min dataset")

	elif cfg.split=='by_tcn_30':
		print('Splitting by tcn using 30sec')
		train_files, val_files, test_files = random_split_df('files/datasets/30secdataset.csv', manual_seed=cfg.random_state)
		train_set =  ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=train_files)
		val_set = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=val_files)
		test_set = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=test_files)
		print(f"There are {len(train_set)} samples in the training dataset, {len(val_set)} in the val set and {len(test_set)} in the test set. Split made according to random split on 30sec dataset")

	else:
		raise ValueError(f"Unsupported dataset split: {cfg.split}")
	
	return train_set, val_set, test_set

def split_dataset(dataset, cfg, proportions=[0.7, 0.15, 0.15]):   
	"""Split Dataset into training and validation sets.

	Args:
		dataset (Dataset): Dataset to split.
		cfg (omegaconf.dictconfig.DictConfig): Config dictionary.
		proportions (list, optional): Proportions of training, validation and test sets when randomly splitting. Defaults to [0.7, 0.15, 0.15].

	Raises:
		ValueError: _description_

	Returns:
		tuple: _description_
	"""
	if isinstance(dataset, SpermWhaleDataset):
		return split_sw_dataset(dataset, cfg, proportions)
	elif isinstance(dataset, ClickDataset):
		return split_click_dataset(dataset, cfg, proportions)
	else: 
		raise TypeError(f'{type(dataset)} not valid for dataset to split.')

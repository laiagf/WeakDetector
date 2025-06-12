# Dataset management functions
import pandas as pd
pd.options.mode.chained_assignment = None
from weakDetector.datasets.spermWhaleDataset import SpermWhaleDataset
from weakDetector.datasets.clickDataset import ClickDataset
from weakDetector.utils.mm import get_target_length

import os
import torch

from weakDetector.config import ROOT_DIR, SOURCES


def random_split_df(csv_path, train_proportion = 0.7, manual_seed=0):
	"""Get filenames after random split of df.

	Args:
		csv_path (string): path to 
		train_proportion (float, optional): _description_. Defaults to 0.7.
		manual_seed (int, optional): _description_. Defaults to 0.

	Returns:
		_type_: _description_
	"""
	df = pd.read_csv(os.path.join(ROOT_DIR, csv_path)) # TODO how do we handle root dir here
	total_size = len(df)
	# Calculate sizes of the train and validation sets
	train_size = int(train_proportion * total_size)

	# Generate the same random indices as before
	indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(manual_seed))

	# Use the same indices to split into train and validation sets
	train_indices = indices[:train_size].tolist()
	train_files = list(set(df.File[train_indices]))

	val_indices = indices[train_size:].tolist()
	val_files = list(set(df.File[val_indices]))

	return train_files, val_files


# TODO rethink this this needs to be in the class itself or something like that
def split_sw_dataset(dataset:SpermWhaleDataset, cfg, random_prop = 0.7):
	"""Split SpermWhaleDataset into training and validation sets.

	Args:
		dataset (SpermWhaleDataset): Dataset to split.
		cfg (omegaconf.dictconfig.DictConfig): Config dictionary.
		random_prop (float, optional): Proportion of training set when randomly splitting. Defaults to 0.7.

	Raises:
		ValueError: _description_

	Returns:
		tuple: _description_
	"""
	if 'split' not in cfg.keys() or cfg['split']=='random':
		split_n=int(len(dataset)*random_prop) 
		train_set, val_set = torch.utils.data.random_split(dataset, [split_n, len(dataset)-split_n], generator=torch.Generator().manual_seed(cfg.random_state)) 
		df_dataset = dataset.annotations
		df_dataset['training']=0
		df_dataset.training[train_set.indices]=1

	elif cfg['split']=='by_source':
		train_set = dataset
		val_set_sources = [s for s in SOURCES if s not in cfg['train_sources']]
		print('Val sources', val_set_sources)

		val_set = SpermWhaleDataset(annotations_file=cfg.annotations_file,
								files_dir=dataset.files_dir,
								target_length=get_target_length(cfg),
								sources=val_set_sources, min_snr=cfg.min_snr)
							  #  channels=cfg.channels) TODO
		
		df_train = train_set.annotations
		df_val = val_set.annotations
		df_train['training'] = 1
		df_val['training'] = 0
		df_dataset = pd.concat([df_train, df_val]).reset_index(drop=True)
	
	else:
		raise ValueError(f"Unsuported dataset split method {cfg['split']}.")
	
	return train_set, val_set, df_dataset


def split_click_dataset(dataset:ClickDataset, cfg, random_prop=0.7):
	"""Split ClickDataset into training and validation sets.

	Args:
		dataset (ClickDataset): Dataset to split.
		cfg (omegaconf.dictconfig.DictConfig): Config dictionary.
		random_prop (float, optional): Proportion of training set when randomly splitting. Defaults to 0.7.

	Raises:
		ValueError: _description_

	Returns:
		tuple: _description_
	"""
	if 'split' not in cfg.keys() or cfg['split']=='random' or cfg['train_sources']=='all':
		split_n=int(len(dataset)*random_prop) 
		train_set, val_set = torch.utils.data.random_split(dataset, [split_n, len(dataset)-split_n],  generator=torch.Generator().manual_seed(cfg.random_state)) 
		print(f"There are {len(dataset)} samples in the dataset, random split ({len(train_set)} training / {len(val_set)} validation).")
	
	elif cfg.split=='by_source':
		train_set=dataset
		val_sources = [s for s in SOURCES if s not in cfg.train_sources]
		val_set = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=val_sources)
		print(f"There are {len(train_set)} samples in the training dataset (sources{cfg.train_sources}, and {len(val_set)} in the validation set (sources {val_sources})")
	
	elif cfg.split=='by_tcn_4':
		train_files, val_files = random_split_df('files/datasets/4mindatasetB_LaiasVersion.csv', manual_seed=cfg.random_state)
		train_set =  ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=train_files)
		val_set = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=val_files)
		print(f"There are {len(train_set)} samples in the training dataset  and {len(val_set)}. Split made according to random split on 4min dataset B")

	elif cfg.split=='by_tcn_30':
		print('Splitting by tcn using 30secC_LaiasVersion')
		train_files, val_files = random_split_df('files/datasets/30secdatasetC_LaiasVersion.csv', manual_seed=cfg.random_state)
		train_set =  ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=train_files)
		val_set = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, sources=cfg.train_sources, split_files=val_files)
		print(f"There are {len(train_set)} samples in the training dataset  and {len(val_set)}. Split made according to random split on 30 sec dataset C.")

	else:
		raise ValueError(f"Unsupported dataset split: {cfg.split}")
	
	return train_set, val_set

def split_dataset(dataset, cfg, random_prop=0.85):   
	"""Split Dataset into training and validation sets.

	Args:
		dataset (Dataset): Dataset to split.
		cfg (omegaconf.dictconfig.DictConfig): Config dictionary.
		random_prop (float, optional): Proportion of training set when randomly splitting. Defaults to 0.7.

	Raises:
		ValueError: _description_

	Returns:
		tuple: _description_
	"""
	if isinstance(dataset, SpermWhaleDataset):
		return split_sw_dataset(dataset, cfg, random_prop)
	elif isinstance(dataset, ClickDataset):
		return split_click_dataset(dataset, cfg, random_prop)
	else: 
		raise TypeError(f'{type(dataset)} not valid for dataset to split.')

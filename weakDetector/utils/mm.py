# Model management functions
import pandas as pd
import torch
from omegaconf import OmegaConf
import os
from tqdm import tqdm

from weakDetector.models.vae_1d import VAE_1D
from weakDetector.models.vae_resnet import VAE_ResNet
from weakDetector.config import ROOT_DIR, SOURCES
from weakDetector.utils.func import standardise, renormalise

def load_vae(cfg, length=None, device='cpu'):
	"""Load ae model.

	Args:
		cfg (omegaconf.dictconfig.DictConfig): _description_
		length (int): Length of extracted features

	Returns:
		nn.Module: Model
	"""
	
	model_name = cfg.model.name
	if model_name == 'vae_1d':
		model = VAE_1D(length, cfg.model.n_classes, cfg.model.latent_size, cfg.model.out_channels, device)		
	if model_name == 'vae_resnet':
		model =  VAE_ResNet(int(cfg.model.latent_size))
	return model


def get_target_length(cfg):
	dataset_name = cfg['dataset']

	if dataset_name=='long_wf':
		length=2048
	if dataset_name=='short_wf' or dataset_name=='spectral_profile':
		length=512		
	if dataset_name=='spectrogram':
		length = 512*256
	return int(cfg['target_seconds']*48000/length)

def find_length(cfg):
	dataset_name = cfg['dataset']
	print(dataset_name)
	if dataset_name=='long_wf':
		tensor_shape = [1, 1, 2048]
		length=2048
	if dataset_name=='short_wf':
		tensor_shape = [1, 1, 512]
		length=512		
	if dataset_name=='spectral_profile':
		length=200
		tensor_shape=[1, 1, 200]
	if dataset_name=='spectrogram':
		length = -1
		tensor_shape = [1, 1, 128, 128]
	return length, tensor_shape



def get_embedding_standardisation(run_path, device):
	

	cfg_ae_path = os.path.join(run_path, '.hydra/config.yaml')
	cfg_ae = OmegaConf.load(cfg_ae_path)

	latent_size = cfg_ae['model']['latent_size']*2
	dataset = cfg_ae['dataset']

	ae_input_length, tensor_shape = find_length(cfg_ae)

	model = load_vae(cfg_ae, ae_input_length).to(device)
	model.load_state_dict(torch.load(os.path.join(run_path,'trained_vae.pth') , map_location=device), strict=False)
	model.eval()

	df = pd.read_csv(ROOT_DIR+'/files/short_clips.csv')
	if cfg_ae['train_sources']!='all':
		df = df[df.Dataset.isin(cfg_ae['train_sources'])]
		df = df.reset_index(drop=True)

	# only training clips
	# Generate the indices for splitting
	split_n = int(len(df)*0.85)
	generator = torch.Generator().manual_seed(cfg_ae.random_state)
	indices = torch.randperm(len(df), generator=generator).tolist()  # Shuffle indices deterministically
	# Split indices
	train_indices = indices[:split_n]
	df = df[df.index.isin(train_indices)].reset_index(drop=True)



	tensor_dir = cfg_ae.tensor_dir

	dfs = []

	#sources = list(set(df.Dataset))
	sources = SOURCES
	for s in sources:
		embeddings = []
		for i in tqdm(df.index[df.Dataset==s], desc='standardising'):
			t = torch.load(tensor_dir+df.Tensor_name[i])
			t = torch.nan_to_num(t)		

			

			#if cfg_ae['dataset']=='spectral_profile':
			#	t = moving_average(t, 8)
			#	t = torch.from_numpy(t)
			#t = standardise(t)

			if cfg_ae.dataset=='long_wf' or cfg_ae.dataset=='short_wf' or ('scale' in cfg_ae.keys() and cfg_ae.scale=='standardise'):
				t = standardise(t)
			else:
				t = renormalise(t)
			
			t = t.reshape(tensor_shape)

			with torch.no_grad(): 
				z, mu, logvar = model.encoder(t.float().to(device))

			embedding = torch.concat((mu, logvar), 1)

			embeddings.append(embedding.cpu()[0, :])


		c = torch.stack(embeddings, dim=1)
		c_clipped = torch.clamp(c, min=-1000, max=1000)

		means = []
		stds = []
		for i in range(latent_size):
			c_i = c_clipped[i, :]
			mean = c_i.mean()
			std = c_i.std()
			means.append(mean.item())
			stds.append(std.item())
			
		df_means_stds = pd.DataFrame({'Dataset':s, 'dimension':[i for i  in range(latent_size)], 'row_mean': means, 'row_std':stds})
		dfs.append(df_means_stds)


	df = pd.concat(dfs, axis=0)
	df.to_csv(os.path.join(run_path,'standard_dict.csv'), index=False)

	return 







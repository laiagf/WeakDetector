
import pandas as pd
import os
import torch
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore")

from weakDetector.utils.mm import load_vae
from weakDetector.core.featureEngines import VAEFeatureExtractor

from weakDetector.config import ROOT_DIR, WAV_PATH

from joblib import Parallel, delayed
from tqdm import tqdm

def process_file(feat_extractor, f, wav_dir, out_dir):
	if not os.path.exists(out_dir+f[:-3]+'pt'):
		seq = feat_extractor(os.path.join(wav_dir, f))    
		torch.save(seq, out_dir+f[:-3]+'pt')

def extract_embeddings(vae_dir, device, wavfile_length=4*60):



	window_lengths = {'long_wf':2048, 'short_wf':512, 'spectrogram':128*256, 'spectral_profile':512}
	feature_lengths = {'long_wf':2048, 'short_wf':512, 'spectrogram':-1, 'spectral_profile':200}

	cfg = OmegaConf.load(os.path.join(vae_dir, '.hydra/config.yaml'))

	# TODO - have this function in utils
	model = load_vae(cfg, feature_lengths[cfg.dataset])
	model.load_state_dict(torch.load(os.path.join(vae_dir,'trained_vae.pth'), map_location=device ), strict=False)

	feat_extractor = VAEFeatureExtractor(window_size=window_lengths[cfg.dataset], 
									  latent_size=cfg.model.latent_size, 
									  input_type=cfg.dataset, 
									  model=model,
									  target_length=wavfile_length*48000, n_parallel=1)
	
	# TODO Improve csv links and names
	if wavfile_length==4*60:
		df = pd.read_csv(os.path.join(ROOT_DIR, 'files/4minDataset.csv'))
	elif wavfile_length==30:
		df = pd.read_csv(os.path.join(ROOT_DIR, 'files/30secDataset.csv'))
	else:
		raise ValueError(f'Value {wavfile_length} for wavfile length not accepted')
	
	out_dir = os.path.join(vae_dir, f'embeddings/{wavfile_length}/')
	if not os.path.exists(os.path.join(vae_dir, 'embeddings')):
		os.mkdir(os.path.join(vae_dir, 'embeddings'))
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)



	#Parallel(n_jobs=8)(delayed(process_file)(feat_extractor, f, WAV_PATH, out_dir)  for f in list(df.FileName))
	Parallel(n_jobs=8)(
		delayed(process_file)(feat_extractor, f, WAV_PATH, out_dir) 
		for f in tqdm(list(df.FileName))
	)
	#for k, f in enumerate(df.FileName):




if __name__=='__main__':
	if torch.cuda.is_available():
		device="cuda:0"
	else:
		device="cpu"
	print('eo')
	# Find all directories with trained models
	vae_paths = []
	#TODO improve this path
	for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT_DIR, "experiments/experiment_VAE/train_vae/run_outputs/")):
	#for  dirpath, dirnames, filenames in os.walk(os.path.join(ROOT_DIR, "experiments/experiment_VAE/train_vae/run_outputs_old/dataset=spectrogram,split=random,train_sources=all/")):

		#for filename in [f for f in filenames if f.endswith(".log")]:
		if 'trained_vae.pth' in filenames:
			vae_paths.append(dirpath)
	#print(vae_paths)
	#vae_paths = ['/home/laia/Projects/WeakDetector/experiments/experiment_VAE/train_vae/run_outputs/dataset=short_wf,model.out_channels=8,scale=standardise/random_split,sources=all/24/random_state=1/']
	vae_paths = ['/mnt/spinning1/WeakDetector/experiments/experiment_VAE/train_vae/run_outputs/dataset=spectrogram/random_split,sources=all/64/random_state=1/']
	for vae_path in vae_paths:
		print(vae_path)
		extract_embeddings(vae_path, device)
		#extract_embeddings(vae_path, device, wavfile_length=30)


	
	

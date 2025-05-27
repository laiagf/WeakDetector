import pandas as pd
pd.options.mode.chained_assignment = None
from omegaconf import OmegaConf
import hydra

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from weakDetector.models.tcn import TCN
from weakDetector.datasets.spermWhaleDataset import SpermWhaleDataset
from weakDetector.core.trainers import ClassifierTrainer
from weakDetector.utils.dm import split_dataset
from weakDetector.config import ROOT_DIR, SOURCES, CODE_DIR




@hydra.main(config_path=CODE_DIR+"experiments/experiment_VAE/config", config_name="config.yaml",version_base=None)
def main(cfg):

	if torch.cuda.is_available():
		device="cuda:0"
	else:
		device="cpu"
	print(f"using {device} device")

	
	# Load Dataset
	vae_run_path = os.path.join(ROOT_DIR, f'experiments/experiment_VAE/train_vae/run_outputs/dataset={cfg.dataset}/{cfg.split}_split,sources={cfg.train_sources}/{cfg.latent_size}/random_state={cfg.vae_random_state}/') 	
	cfg_vae_path = vae_run_path +'.hydra/config.yaml'
	cfg_vae = OmegaConf.load(cfg_vae_path)


	model = TCN(cfg_vae.model.latent_size, cfg.model.output_size, [cfg.model.n_hid]*cfg.model.levels, 
		kernel_size=cfg.model.kernel_size, dropout=cfg.model.dropout)
	model.to(device)

	if cfg.standard:
		df_standard = pd.read_csv(os.path.join(vae_run_path, 'standard_dict.csv'))
	else:
		df_standard=None

	dataset = SpermWhaleDataset(annotations_file=cfg.annotations_file,
								files_dir=os.path.join(vae_run_path, 'embeddings/'+str(cfg.target_seconds)),
								target_length=cfg.target_length, ## TODO homogenise this
								sources=cfg.train_sources, min_snr=cfg.min_snr, df_standard=df_standard,
								channels=[i for i in range(cfg_vae.model.latent_size)])

	# split datasets
	train_set, val_set, df_dataset = split_dataset(dataset, cfg)
	

	print(f"AE train sources were {cfg_vae.train_sources} and AE split was {cfg_vae.split}")


	# dataloader
	train_loader = DataLoader(dataset=train_set, batch_size=cfg.model.batch_size, shuffle=True)
	val_loader = DataLoader(dataset=val_set, batch_size=cfg.model.batch_size, shuffle=True)



	optimiser = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
	trainer = ClassifierTrainer(model=model, optimiser=optimiser, lr=cfg.model.lr,
								loss_func = F.nll_loss, lr_decrease_rate=cfg.model.decrease_rate)
	

	trainer(train_loader, val_loader, cfg.model.n_epochs, device)


	hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
	outputdir = hydra_cfg['runtime']['output_dir']
	torch.save(trainer.model.state_dict(), os.path.join(outputdir, 'trained_tcn.pth'))	
 
	df_log = trainer.training_log
	df_log.to_csv(os.path.join(outputdir, 'training_log.csv'), index=False)
	df_dataset.to_csv(os.path.join(outputdir, 'dataset.csv'))


if __name__=='__main__':
	main()
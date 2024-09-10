import pandas as pd
from omegaconf import OmegaConf
import hydra

import os

import torch
from torch.utils.data import DataLoader

from models.tcn import TCN
from datasets.spermWhaleDataset import SpermWhaleDataset
from classes.trainers import ClassifierTrainer
from utils.dm import split_dataset
from config import ROOT_DIR, SOURCES




def main(cfg):

	if torch.cuda.is_available():
		device="cuda:0"
	else:
		device="cpu"
	print(f"using {device} device")

	
	# Load Dataset
	vae_run_path = os.path.join(ROOT_DIR, 'experiments/experiment_VAE/train_vae/run_outputs/') + cfg.run_path 	
	cfg_vae_path = vae_run_path +'.hydra/config.yaml'
	cfg_vae = OmegaConf.load(cfg_vae_path)


	model = TCN(cfg.n_channels, cfg.parameters.output_size, [cfg.parameters.n_hid]*cfg.parameters.levels, 
		kernel_size=cfg.parameters.kernel_size, dropout=cfg.parameters.dropout)
	model.to(device)

	dataset = SpermWhaleDataset(annotations_path=cfg.annotations_path,
                                files_dir=cfg.files_dir,
                                target_length=cfg.target_length,
                                sources=cfg.sources,
                                channels=cfg.channels)

    # split datasets
	train_set, val_set, df_dataset = split_dataset(dataset, cfg)
	

	print(f"AE train sources were {cfg_vae.train_sources} and AE split was {cfg_vae.split}")


	# dataloader
	train_loader = DataLoader(dataset=train_set, batch_size=cfg.model.batch_size, shuffle=True)
	val_loader = DataLoader(dataset=val_set, batch_size=cfg.model.batch_size, shuffle=True)



	optimiser = torch.optim.Adam(model.parameters(), lr=cfg.parameters.lr)
	trainer = ClassifierTrainer(model=model, optimiser=optimiser, lr=cfg.parameters.lr,
                                loss_func = F.nll_loss, lr_decrease_rate=10)
    

	trainer(train_loader, val_loader, cfg.parameters.n_epochs, device)


	hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
	outputdir = hydra_cfg['runtime']['output_dir']
	torch.save(trainer.model.state_dict(), os.path.join(outputdir, 'trained_tcn.pth'))	
 
	df_log = trainer.training_log
	df_log.to_csv(os.path.join(outputdir, 'training_log.csv'), index=False)
	df_dataset.to_csv(os.path.join(outputdir, 'dataset.csv'))

import torch
from torch.utils.data import DataLoader

import os
import hydra
import pandas as pd


from weakDetector.datasets.clickDataset import ClickDataset

from weakDetector.utils.dm import split_dataset
from weakDetector.utils.mm import load_vae
from weakDetector.core.trainers import AETrainer




from weakDetector.config import ROOT_DIR


torch.manual_seed(0)




@hydra.main(config_path=ROOT_DIR+"/experiments/experiment_VAE/train_vae/config", config_name="config.yaml",version_base=None)
def main(cfg):

	if torch.cuda.is_available():
		device="cuda:0"
	else:
		device="cpu"

	print(f"using {device} device")
	


	if cfg.dataset=='long_wf' or cfg.dataset=='short_wf' or cfg.scale=='standardise': 
		scale_method = 'standardise'
	else:
		scale_method='normalise'

	print(f"training {cfg.model.name} on dataset {cfg.dataset} using scale method {scale_method} with latent space of dim {cfg.model.latent_size}. Using {scale_method} scaling method.")



	dataset = ClickDataset(cfg.dataset, cfg.csv_file, cfg.tensor_dir, scale_method=scale_method, sources=cfg.train_sources, min_snr=cfg.min_snr)


	# Split dataset into training and validation files
	train_set, val_set = split_dataset(dataset, cfg)


	train_loader = DataLoader(dataset=train_set, batch_size=cfg.model.parameters.batch_size, shuffle=True)
	val_loader = DataLoader(dataset=val_set, batch_size=cfg.model.parameters.batch_size, shuffle=True)


	model = load_vae(cfg, dataset.feature_length).to(device)


	# TODO include ae params in config... change from cfg.model.parameters.lr to cfg.ae_model.parameters.lr
	optimiser = torch.optim.Adam(model.parameters(), lr=cfg.model.parameters.lr)


	trainer = AETrainer(model, optimiser, cfg.model.parameters.lr, lr_decrease_rate=100)

	hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
	print(hydra_cfg['runtime']['output_dir'])
	outputdir = hydra_cfg['runtime']['output_dir']
	

	trainer(train_loader, val_loader, cfg.model.parameters.n_epochs, device, outpath=outputdir, checkpoints_every=100)


	


	
	torch.save(model.state_dict(), os.path.join(outputdir, 'trained_vae.pth'))	
	df_log = trainer.training_log
	df_log.to_csv(os.path.join(outputdir, 'training_log.csv'), index=False)


	return 

if __name__=='__main__':
    main()

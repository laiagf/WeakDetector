import hydra
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import os
from models.tcn import TCN
from datasets.spermWhaleDataset import SpermWhaleDataset
from classes.trainers import ClassifierTrainer
from utils.dm import split_dataset

@hydra.main(config_path="config", config_name="config.yaml",version_base=None)
def main(cfg):
    
    # TODO maybe rethink thsi (put it elsewhere)
    if torch.cuda.is_available():
        device="cuda:0"
    else:
        device="cpu"
    print(f"using {device} device")

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
    
	# dataloader
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.parameters.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=cfg.parameters.batch_size, shuffle=True)

    
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
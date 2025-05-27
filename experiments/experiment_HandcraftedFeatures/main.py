import hydra
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import os
from weakDetector.models.tcn import TCN
from weakDetector.datasets.spermWhaleDataset import SpermWhaleDataset
from weakDetector.core.trainers import ClassifierTrainer
from weakDetector.utils.dm import split_dataset
from weakDetector.config import CODE_DIR, DATA_PATH

@hydra.main(config_path=CODE_DIR+"experiments/experiment_HandcraftedFeatures/config", config_name="config.yaml",version_base=None)
def main(cfg):
    # TODO maybe rethink thsi (put it elsewhere)
    if torch.cuda.is_available():
        device="cuda:0"
    else:
        device="cpu"
    print(f"using {device} device")


    model = TCN(cfg.n_channels, cfg.model.output_size, [cfg.model.n_hid]*cfg.model.levels, 
		kernel_size=cfg.model.kernel_size, dropout=cfg.model.dropout)
    model.to(device)

    
    files_dir = os.path.join(DATA_PATH, f'{cfg.features}_Vectors/')
    if cfg.features=='RMS':
        files_dir = os.path.join(files_dir, f'RMS_{cfg.resolution}_{cfg.n_channels}band')
        if cfg.n_channels>1:
            files_dir +='s'

    if cfg.resolution=='HR': 
        window_size=512
    elif cfg.resolution=='LR':
        window_size=2048

    target_length = int(cfg.target_seconds*48000/window_size)


    dataset = SpermWhaleDataset(annotations_file=cfg.annotations_file,
                                files_dir=files_dir,
                                target_length=target_length,
                                sources=cfg.train_sources,
                                channels='all', min_snr=cfg.min_snr)

    # split datasets
    train_set, val_set, df_dataset = split_dataset(dataset, cfg)
    
	# dataloader
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=cfg.model.batch_size, shuffle=True)

    
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    trainer = ClassifierTrainer(model=model, optimiser=optimiser, lr=cfg.model.lr,
                                loss_func = F.nll_loss, lr_decrease_rate=10)
    
    trainer(train_loader, val_loader, cfg.model.n_epochs, device)


    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    outputdir = hydra_cfg['runtime']['output_dir']
    torch.save(trainer.model.state_dict(), os.path.join(outputdir, 'trained_tcn.pth'))	
 
    df_log = trainer.training_log
    df_log.to_csv(os.path.join(outputdir, 'training_log.csv'), index=False)
    df_dataset.to_csv(os.path.join(outputdir, 'dataset.csv'))


if __name__=='__main__':
    main()
# Model management functions
from weakDetector.models.vae_1d import VAE_1D
from weakDetector.models.vae_resnet import VAE_ResNet


def load_vae(cfg, length=None):
    """Load ae model.

    Args:
        cfg (omegaconf.dictconfig.DictConfig): _description_
        length (int): Length of extracted features

    Returns:
        nn.Module: Model
    """
	
    model_name = cfg.model.name
    if model_name == 'vae_1d':
        model = VAE_1D(length, cfg.model.n_classes, cfg.model.latent_size, cfg.model.out_channels)		
    if model_name == 'vae_resnet':
        model =  VAE_ResNet(int(cfg.model.latent_size))
    return model

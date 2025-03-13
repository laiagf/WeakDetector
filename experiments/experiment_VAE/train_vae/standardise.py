from weakDetector.utils.mm import get_embedding_standardisation
from weakDetector.config import ROOT_DIR
import sys
import os
#run_dir = sys.argv[1]

if __name__=='__main__':
    runs_path = os.path.join(ROOT_DIR,'experiments/experiment_VAE/train_vae/run_outputs/')
    datasets_path = [os.path.join(runs_path, d) for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d) )]
    sources_paths = []
    for p in datasets_path:
        sources_paths += [os.path.join(p, s) for s in os.listdir(p) if os.path.isdir(os.path.join(p, s) )]
    latent_paths = []
    for p in sources_paths:
        latent_paths += [os.path.join(p, l) for l in os.listdir(p) if os.path.isdir(os.path.join(p, l) )]
    
    rs_paths = []
    for p in latent_paths:
        rs_paths += [os.path.join(p, r) for r in os.listdir(p) if os.path.isdir(os.path.join(p, r) )]
    #run_path = os.path.join(ROOT_DIR,'experiments/experiment_VAE/train_vae/run_outputs/'+run_dir)
    for run_path in rs_paths:
        if ('trained_vae.pth' in os.listdir(run_path)) and ('standard_dict.csv' not in os.listdir(run_path)):
            get_embedding_standardisation(run_path, 'cuda')

from weakDetector.utils.mm import get_embedding_standardisation
from weakDetector.config import ROOT_DIR
import sys
import os
run_dir = sys.argv[1]

if __name__=='__main__':

    run_path = os.path.join(ROOT_DIR,'experiments/experiment_VAE/train_vae/run_outputs/'+run_dir)
    get_embedding_standardisation(run_path, 'cuda')

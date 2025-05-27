from weakDetector.core.featureEngines import HeuristicFeatureExtractor

import pandas as pd
import os
import torch
from weakDetector.config import ROOT_DIR, WAV_PATH, DATA_PATH
import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
def extract_rms():
    """Extract all RMS-only feature vectors.
    """

    resolutions = {'HR': 512, 'LR':2048}

    frequency_bands = {
        '1band': [1000, 20000],
        '3bands': [1000, 6000, 12000, 20000],
        '5bands': [1000, 2000, 4000, 8000, 16000, 20000]
    }

    df = pd.read_csv(os.path.join(ROOT_DIR, 'files/4minDataset.csv'))
    out_path = os.path.join(DATA_PATH, 'RMS_Vectors/')

    for res in resolutions.keys():
        for fs in frequency_bands.keys():
            out_dir = os.path.join(out_path, f'RMS_{res}_{fs}/')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            feat_extractor = HeuristicFeatureExtractor(window_size=resolutions[res],
                                rms=True, rms_freqs=frequency_bands[fs])
            
            for f in df.FileName:
                if not os.path.exists(out_dir+f[:-3]+'pt'):

                    seq = feat_extractor(os.path.join(WAV_PATH,f))    
                    torch.save(seq, out_dir+f[:-3]+'pt')


def extract_spectral_file(feat_extractor, f, out_dir):
    print(f'{datetime.datetime.now()}: Processing file {f}')
    if not os.path.exists(out_dir+f[:-3]+'pt'):
        seq = feat_extractor(os.path.join(WAV_PATH,f))    
        torch.save(seq, out_dir+f[:-3]+'pt')

def extract_spectral():
    """Extract all spectral parameters feature vectors.
    """

    resolutions = {'HR': 512,  'LR':2048}


    df = pd.read_csv(os.path.join(ROOT_DIR, 'files/4minDataset.csv'))
    out_path = os.path.join(DATA_PATH, 'Spectral_Vectors/')

    for res in resolutions.keys():
            out_dir = out_path
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            feat_extractor = HeuristicFeatureExtractor(window_size=resolutions[res],
                                rms=True, rms_freqs=[1000, 20000], mean_freq=True,
                                peak_freq=True, energy_sums=True, spectral_width=True)
            
            #for f in df.FileName:
            
            Parallel(n_jobs=16)(
                delayed(extract_spectral_file)(feat_extractor, f, out_dir) 
                for f in tqdm(list(df.FileName))
            )


if __name__=='__main__':
    #extract_rms()
    extract_spectral()
                


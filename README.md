# WeakDetector

This is the repository for the paper **Variational autoencoders stabilise TCN performance when classifying weakly labelled bioacoustics data** (arXiv:2410.17006).

## Structure
`weakDetector` includes all the necessary code for training and deploying both autoencoders and Temporal Convolutional Networks (TCN) that work on embeddings in the form of a python package.

`Files/`: Contains CSV files with annotations for training and evaluation for the datasets used in our paper. Acompanying audio data is available on demand.

`Experiments/`: Contains a practical implementation of the weakDetector module. As per our paper. this includes compressing 4-minute audio segments and classifying them based on the presence/absence of sperm whale click trains.

## How to find your own sperm whales? 
Tutorial coming very soon

## Paper abstract
Passive acoustic monitoring (PAM) data is often weakly labelled, audited at the scale of detection presence or absence on timescales of minutes to hours. Moreover, this data exhibits great variability from one deployment to the next, due to differences in ambient noise and the signals across sources and geographies. This study proposes a two-step solution to leverage weakly annotated data for training Deep Learning (DL) detection models. Our case study involves binary classification of the presence/absence of sperm whale (*Physeter macrocephalus*) click trains in 4-minute-long recordings from a dataset comprising diverse sources and deployment conditions to maximise generalisability. We tested methods for extracting acoustic features from lengthy audio segments and integrated Temporal Convolutional Networks (TCNs) trained on the extracted features for sequence classification. For feature extraction, we introduced a new approach using Variational AutoEncoders (VAEs) to extract information from both waveforms and spectrograms, which eliminates the necessity for manual threshold setting or time-consuming strong labelling. For classification, TCNs were trained separately on sequences of either VAE embeddings or handpicked acoustic features extracted from the waveform and spectrogram representations using classical methods, to compare the efficacy of the two approaches. The TCN demonstrated robust classification capabilities on a validation set, achieving accuracies exceeding 85% when applied to 4-minute acoustic recordings. Notably, TCNs trained on handpicked acoustic features exhibited greater variability in performance across recordings from diverse deployment conditions, whereas those trained on VAEs showed a more consistent performance, highlighting the robust transferability of VAEs for feature extraction across different deployment conditions.



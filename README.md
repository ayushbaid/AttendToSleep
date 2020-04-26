# AttendToSleep
Sleep Stage Classification using CNN + Attention mechanism (Transformers)



We use the EEG Fpz-Cz single channel data to classify the sleep



## Setup

- TODO: add data download instructions (put processed npy files in data/all/processed)
- Use the environment.yml file to install the conda environment
- Move to directory code by running ```cd code```
- Install the repository by running ```pip install -e .```
- Download the PSG and hypnogram files from PhysioNet by running ```cd data```, ```chmod +x download_physionet.sh```,```./download_physionet.sh```
- Extract signals from EEG Fpz-Cz, EEG Pz-Oz, and EOG horizontal by running ```python prepare_physionet.py --data_dir data```
- Data processing by running ```python python data_processing.py --data_dir data```
- train the cnn-only model by running ```python helper/runner_cnn.py```
- train the final model by running ```python helper/runner_cnn_seq2seq.py```

## Datasets

We are using the expanded Sleep-EDF database from https://physionet.org/content/sleep-edfx/1.0.0/

### Channels
- **EEG Fpz-Cz**
- **EEG Pz-Oz**
- **EOG horizontal**
- Resp oro-nasal
- EMG submental
- Temp rectal
- Event marker

### ETL Pipeline
1. Extract different channel signals from downloaded PSG for each patient, and save as a CSV file. Extract corresponding labels from the hypnogram for the same patient.
2. Load the label CSV and the signal CSV into Spark (PySpark). 
3. Remove unknown labels, only keep stages W, N1, N2, N3, N4, and REM.
4. Segment the signals into 30s epochs.
5. Save the channel signals and corresponding label as NPZ file for models.

### Deep Learning Pipeline
1. CNN
2. Attention on RNN
3. GAN?

## Existing Work

1. Automatic Sleep Stage Scoring Using Time-Frequency Analysis and Stacked Sparse Autoencoders, 2015. [Paper](https://www.ncbi.nlm.nih.gov/pubmed/26464268) 
2. Automatic Sleep Stage Scoring with Single-Channel EEG Using Convolutional Neural Networks, 2016. [Paper](https://arxiv.org/pdf/1610.01683.pdf)
3. Learning Sleep Stages from Radio Signals: A Conditional Adversarial Architecture, 2017. [Paper](http://proceedings.mlr.press/v70/zhao17d/zhao17d.pdf)
4. SLEEPNET: Automated Sleep Staging System via Deep Learning, 2017. [Paper](https://arxiv.org/pdf/1707.08262.pdf)
5. SLEEPER: interpretable Sleep staging via Prototypes fromExpert Rules, 2019. [Paper](https://arxiv.org/pdf/1910.06100.pdf)
6. Neural network analysis of sleep stages enablesefficient diagnosis of narcolepsy, 2018. [Paper](https://www.nature.com/articles/s41467-018-07229-3.pdf) [Code](https://github.com/Stanford-STAGES/stanford-stages)
7. SleepEEGNet: Automated Sleep Stage Scoring with Sequence to Sequence Deep Learning Approach, 2019. [Paper](https://arxiv.org/pdf/1903.02108.pdf) [Code](https://github.com/SajadMo/SleepEEGNet)


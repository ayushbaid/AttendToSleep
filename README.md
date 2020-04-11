# AttendToSleep
Sleep Stage Classification using Attention



## Datasets

We are using the expanded Sleep-EDF database from https://physionet.org/content/sleep-edfx/1.0.0/

### Channels
- EEG Fpz-Cz
- EEG Pz-Oz
- EOG horizontal
- Resp oro-nasal
- EMG submental
- Temp rectal
- Event marker

### ETL Pipeline
1. Separating training vs. testing dataset (randomly/think of the best way to sample while avoiding subsampling on the same sequence). 
2. Compute spectogram in the pipeline.

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


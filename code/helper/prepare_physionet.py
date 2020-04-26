'''
This code is adopted from from https://github.com/akaraspt/deepsleepnet

Author: Ayush Baid, Yujia Xie
'''

import argparse
import glob
import math
import ntpath
import os
import shutil

import csv

from datetime import datetime

import numpy as np

from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf

import dhedfreader as dhedfreader

EPOCH_SEC_SIZE = 30

outputDict = {
    "EEG Fpz-Cz": "0",
    "EEG Pz-Oz": "1",
    "EOG horizontal": "2"
}

def process_single_point(psf_fname, ann_fname, args, select_ch):
    raw = read_raw_edf(psf_fname, preload=True, stim_channel=None)
    sampling_rate = raw.info['sfreq']
    print(sampling_rate)

    raw_ch_df = raw.to_data_frame()[select_ch]
    raw_ch_df = raw_ch_df.to_frame()
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))
    raw_ch_df['patientID'] = psf_fname.replace("-PSG.edf", "").split("/")[-1]
    raw_ch_df.index.name = "time"


    # Get raw header
    with open(psf_fname, 'rb') as f:
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
    raw_start_dt = datetime.strptime(
        h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

    # Read annotation and its header
    with open(ann_fname, 'rb') as f:
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())


    csvList = []

    with open(psf_fname.replace("-PSG.edf", "L.csv"), "w") as fout:
        csvWriter = csv.writer(fout)
        csvWriter.writerow(["patientID", "startTime", "time", "stage"])
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            current = onset_sec * sampling_rate
            while current < onset_sec * sampling_rate + duration_sec * sampling_rate:
                csvWriter.writerow([psf_fname.replace("-PSG.edf", "").split("/")[-1], int(onset_sec), int(current), "".join(ann_char).split(" ")[-1]])
                current += 1

    print("start")
    raw_ch_df.to_csv(psf_fname.replace("-PSG.edf", "%sR.csv" % outputDict[select_ch]))
    print("end")

    print("\n=======================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="File path to the CSV file that contains walking data.")
    args = parser.parse_args()

    channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
    for channel in channels:
        select_ch = channel

        # Read raw and annotation EDF files
        psg_fnames = [os.path.abspath(x) for x in glob.glob(
            os.path.join(args.data_dir, "*PSG.edf"))]
        ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
        psg_fnames.sort()
        ann_fnames.sort()
        psg_fnames = np.asarray(psg_fnames)
        ann_fnames = np.asarray(ann_fnames)

        for i in range(len(psg_fnames)):
            try:
                process_single_point(psg_fnames[i], ann_fnames[i], args, select_ch)
            except Exception as e:
                print(str(e))


if __name__ == "__main__":
    main()

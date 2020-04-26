'''
This code is for data processing

Author: Yujia Xie
'''
import findspark
import argparse
import pyspark
from pyspark.sql import SQLContext, SparkSession
from os import listdir
from os.path import isfile, join
import numpy as np
import os


def load(data_dir):
	sc = pyspark.SparkContext("local", "Sleep")
	sql = SparkSession.builder.master("local").appName("SleepSQL").getOrCreate()
	
	# Load data from CSV files
	fpzcz = sql.read.csv("%s/*0R.csv" % data_dir, header="True").rdd
	pzoz = sql.read.csv("%s/*1R.csv" % data_dir, header="True").rdd
	horizontal = sql.read.csv("%s/*2R.csv" % data_dir, header="True").rdd
	label = sql.read.csv("%s/*L.csv" % data_dir, header="True").rdd
	
	# Attach label to raw channel signal data: time, patientID, signal, startTime, label
	fpzcz_label = fpzcz.map(lambda x:((x[0], x[2]), x[1])).join(label.map(lambda x:((x[2], x[0]), (x[1], x[3])))).map(lambda x:(int(x[0][0]), x[0][1], float(x[1][0]), int(x[1][1][0]), x[1][1][1]))
	pzoz_label = pzoz.map(lambda x:((x[0], x[2]), x[1])).join(label.map(lambda x:((x[2], x[0]), (x[1], x[3])))).map(lambda x:(int(x[0][0]), x[0][1], float(x[1][0]), int(x[1][1][0]), x[1][1][1]))
	horizontal_label = horizontal.map(lambda x:((x[0], x[2]), x[1])).join(label.map(lambda x:((x[2], x[0]), (x[1], x[3])))).map(lambda x:(int(x[0][0]), x[0][1], float(x[1][0]), int(x[1][1][0]), x[1][1][1]))

	# Filter out unknown stages
	fpzcz_filtered = fpzcz_label.filter(lambda x: (x[4] == "1" or x[4] == "2" or x[4] == "3" or x[4] == "4" or x[4] == "W" or x[4] == "R"))
	pzoz_filtered = pzoz_label.filter(lambda x: (x[4] == "1" or x[4] == "2" or x[4] == "3" or x[4] == "4" or x[4] == "W" or x[4] == "R"))
	horizontal_filtered = horizontal_label.filter(lambda x: (x[4] == "1" or x[4] == "2" or x[4] == "3" or x[4] == "4" or x[4] == "W" or x[4] == "R"))
	
	# Segment into 30s epochs, frame rate = 100.0, for label data
	label_segmented = fpzcz_filtered.filter(lambda x: ((x[0] - x[3]) % 3000 == 0))

	# Output as dataframe, switch labels, and split the signals into 30s epochs
	patients = [patient for patient in label_segmented.map(lambda x:x[1]).distinct().collect()]
	for patient in patients:
		master = label_segmented.filter(lambda x: x[1] == patient).sortBy(lambda x: x[0])
		output_label = master.map(lambda x: x[4]).collect()
		output_label = vectorize(np.array([label for label in output_label])).astype(int)
		
		maxTime = master.map(lambda x:x[0]).collect()[-1]
		fpzcz_segment = fpzcz_filtered.filter(lambda x:(x[0] < maxTime + 2 and x[1] == patient)).sortBy(lambda x: x[0]).map(lambda x: x[2]).collect()
		output_fpzcz = processSignal([signal for signal in fpzcz_segment])
		pzoz_segment = pzoz_filtered.filter(lambda x:(x[0] < maxTime + 2 and x[1] == patient)).sortBy(lambda x: x[0]).map(lambda x: x[2]).collect()
		output_pzoz = processSignal([signal for signal in pzoz_segment])
		horizontal_segment = horizontal_filtered.filter(lambda x:(x[0] < maxTime + 2 and x[1] == patient)).sortBy(lambda x: x[0]).map(lambda x: x[2]).collect()
		output_horizontal = processSignal([signal for signal in horizontal_segment])
		
		saveFile(data_dir, patient, output_fpzcz, None, None, output_label)
		
# save the files as a numpy archive for model training
def saveFile(data_dir, patient, fpzcz, pzoz, horizontal, label):
	if not os.path.exists("%s/eeg_fpz_cz" % data_dir):
		os.makedirs("%s/eeg_fpz_cz")
	if not os.path.exists("%s/eeg_pz_oz" % data_dir):
		os.makedirs("%s/eeg_pz_oz")
	if not os.path.exists("%s/eog_horizontal" % data_dir):
		os.makedirs("%s/eog_horizontal")

	output_name = patient + ".npz"

	fpzcz_dict = {"x": fpzcz, "y": label}
	pzoz_dict = {"x": pzoz, "y": label}
	horizontal_dict = {"x": horizontal, "y": label}

	np.savez(os.path.join("%s/eeg_fpz_cz" % data_dir, output_name), **fpzcz_dict)
	np.savez(os.path.join("%s/eeg_pz_oz" % data_dir, output_name), **pzoz_dict)
	np.savez(os.path.join("%s/eog_horizontal" % data_dir, output_name), **horizontal_dict)

# represent the stage with numerical values
def vectorize(array):
	array = np.where(array=='W', 0, array)
	array = np.where(array=='1', 1, array)
	array = np.where(array=='2', 2, array)
	array = np.where(array=='3', 3, array)
	array = np.where(array=='4', 3, array)
	array = np.where(array=='R', 4, array)
	return array

# split the signal into 30s echpos
def processSignal(array):
	split = len(array) / 3000
	array = np.array(array).astype(np.float32)
	array = np.asarray(np.split(array, split))
	return array

def script():
	findspark.init()
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="/data/physionet_sleep",
                        help="File path to the CSV file that contains RAW and LABEL data.")
	args = parser.parse_args()
	data_dir = args.data_dir
	load(data_dir)

if __name__ == "__main__":
	script()

import numpy as np
import os
from os import listdir
from os.path import isfile, join
import argparse

def script(data_dir):
	output_train = "%s/train" % data_dir
	output_test = "%s/test" % data_dir
	output_validation = "%s/val" % data_dir
	onlyfiles = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-4:] == ".npz"]
	for aFile in onlyfiles:
		with np.load(aFile) as data:
			output_name = aFile.split("/")[-1]
			x = data['x']
			y = data['y']
			generator = np.random.default_rng()
			train_indices = generator.choice(y.shape[0], int(y.shape[0] * 0.7), replace=False)
			remaining_indices = np.array([i for i in range(y.shape[0]) if i not in train_indices])
			validation_indices = generator.choice(remaining_indices, int(remaining_indices.shape[0] * 0.5), replace=False)
			test_indices = np.array([i for i in range(len(remaining_indices)) if i not in validation_indices])
			yTest = y.take(test_indices, axis=0)
			xTest = x.take(test_indices, axis=0)
			xValidation = x.take(validation_indices, axis=0)
			yValidation = y.take(validation_indices, axis=0)
			yTrain = y.take(train_indices, axis=0)
			xTrain = x.take(train_indices, axis=0)
			
			train_dict = {"x": xTrain, "y": yTrain}
			test_dict = {"x": xTest, "y": yTest}
			validation_dict = {"x": xValidation, "y": yValidation}

			np.savez(os.path.join(output_train, output_name), **train_dict)
			np.savez(os.path.join(output_test, output_name), **test_dict)
			np.savez(os.path.join(output_validation, output_name), **validation_dict)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="data",
						help="File path to the CSV file that contains walking data.")

	args = parser.parse_args()
	data_dir = args.data_dir

	output_train = "%s/train" % data_dir
	output_test = "%s/test" % data_dir
	output_test = "%s/val" % data_dir

	if not os.path.exists(output_train):
		os.makedirs(output_train)
	if not os.path.exists(output_test):
		os.makedirs(output_test)
	if not os.path.exists(output_test):
		os.makedirs(output_test)

	script(data_dir)

from utils.file_load_save import *
from utils.data_processing import *
import os
import numpy as np
import glob

## load the training and validation data
def load_train_data(val_version=21, cross_val = 0, normal_proc = False):
	if val_version==23:
		## load training data val_version: 23
		# reimplement the density map generation codes with python
		# combine cdx2, sox17 and dapi together
		# remove the artifacts in sox17[2,3]
		# gaussian sigma: 6.0
		train_data_folder = os.path.expanduser('~/bio_cell_data/train_data')
		cell_types = ['cdx2', 'sox17', 'dapi']
		X_pool = []
		Y_pool = []
		idx_set = []
		for cell_tp in cell_types:
			ptrn = os.path.join(train_data_folder, cell_tp, '*.pkl')
			file_names = glob.glob(ptrn)
			X_list = []
			Y_list = []
			for fn in file_names:
				X, _, _, Y = read_train_pickles(fn, keys = ['ori', 'rgb', 'annot', 'den'])
				X_list.append(X)
				Y_list.append(Y)
			X_arr = np.array(X_list)
			Y_arr = np.array(Y_list)
			X_pool.append(X_arr)
			Y_pool.append(Y_arr)
			# load the index for each set
			idx_file_name = os.path.join(train_data_folder,cell_tp+'.txt')
			idx_ = []
			with open(idx_file_name) as f:
				for line in f.readlines():
					idx_.append(int(float(line.split()[0])))
			idx_set.append(idx_)
# 		print(idx_set)
		set_len = len(X_pool[0])
		idx = list(range(set_len))
		train_idx = []
		test_idx = []
		if cross_val == 0:
			hlen = int(set_len/2)
			train_idx = idx[0:hlen]
			test_idx = idx[hlen:]
		elif cross_val > 0:
			intvl = int(set_len/5)
			train_idx = idx
			test_idx = idx[(cross_val-1)*intvl:cross_val*intvl]
			for dx in test_idx:
				train_idx.remove(dx)
		X_train = []
		Y_train = []
		X_test = []
		Y_test = []
		print('******** dataset partition summary ******')
		for i in range(len(X_pool)):
			select_idx_set = idx_set[i]
			real_tr_idx = []
			real_te_idx = []
			for ix in train_idx:
				if ix < len(idx_set[i]):
					sl_idx = select_idx_set[ix]
					X_train.append(X_pool[i][sl_idx])
					Y_train.append(Y_pool[i][sl_idx])
					real_tr_idx.append(sl_idx)
			for jx in test_idx:
				if jx < len(idx_set[i]):
					sl_idx = select_idx_set[jx]
					X_test.append(X_pool[i][sl_idx])
					Y_test.append(Y_pool[i][sl_idx])
					real_te_idx.append(sl_idx)
			print('{}\t:{}-{}'.format(cell_types[i],real_tr_idx,real_te_idx))
		X_train = np.array(X_train)
		Y_train = np.array(Y_train)
		X_test = np.array(X_test)
		Y_test = np.array(Y_test)
		train_data = data_normalize(X_train)
		test_data =  data_normalize(X_test)
		train_truths = Y_train
		test_truths = Y_test
		print('****************data loaded! ************')

	if val_version==24:
		## load training data val_version: 24
		# benchmark dataset
		# reimplement the density map generation codes with python
		# gaussian sigma: 3.0	return train_data, test_data, train_truths, test_truths
		train_data_folder = os.path.expanduser('~/dataset/bio_cell_data/train_data')
		data_folder = 'cell_count_benchmark/cells'
		ptrn = os.path.join(train_data_folder, data_folder, '*.pkl')
		file_names = glob.glob(ptrn)
		X_list = []
		Y_list = []
		for fn in file_names:
			X, _, Y = read_pickles(fn, keys = ['cell', 'annot', 'den'])
			X_list.append(X)
			Y_list.append(Y)
		X_arr = np.array(X_list)
		Y_arr = np.array(Y_list)
		if normal_proc:
			X_arr = (X_arr-np.mean(X_arr))/np.std(X_arr)
		nb_train = int(len(X_list)/2)
		X_train = X_arr[:nb_train]
		Y_train = Y_arr[:nb_train]
		X_test = X_arr[nb_train:]
		Y_test = Y_arr[nb_train:]
		train_data = X_train
		test_data =  X_test
		train_truths = Y_train
		test_truths = Y_test

	if val_version==25:
		## load training data val_version: 25
		# benchmark dataset 2
		# reimplement the density map generation codes with python
		# gaussian sigma: 3.0	return train_data, test_data, train_truths, test_truths
		train_data_folder = os.path.expanduser('~/dataset/bio_cell_data/train_data')
		data_folder = 'cell_count_benchmark/BM_dataset_MICCAI2015'
		ptrn = os.path.join(train_data_folder, data_folder, '*.pkl')
		file_names = glob.glob(ptrn)
		X_list = []
		Y_list = []
		for fn in file_names:
			X, _, Y = read_pickles(fn, keys = ['cell', 'annot', 'den'])
			lx = int(X.shape[0]/2)
			X_list.append(X[:lx,:lx,:])
			X_list.append(X[:lx,lx:,:])
			X_list.append(X[lx:,:lx,:])
			X_list.append(X[:lx,:lx,:])
			Y_list.append(Y[:lx,:lx])
			Y_list.append(Y[:lx,lx:])
			Y_list.append(Y[lx:,:lx])
			Y_list.append(Y[:lx,:lx])
		X_arr = np.array(X_list)
		if normal_proc:
			X_arr = (X_arr-np.mean(X_arr))/np.std(X_arr)
		Y_arr = np.array(Y_list)
		nb_train = int(len(X_list)*2/3)
		X_train = X_arr[:nb_train]
		Y_train = Y_arr[:nb_train]
		X_test = X_arr[nb_train:]
		Y_test = Y_arr[nb_train:]
		train_data = X_train
		test_data =  X_test
		train_truths = Y_train
		test_truths = Y_test

	return train_data, test_data, train_truths, test_truths

def load_unannotated_data(data_folder, cell_type = 'cdx2'):
	import os
	data_cell_folder = os.path.join(data_folder, cell_type)
	return load_images_from_folder(data_cell_folder, suffix='.tif', normalized = False)
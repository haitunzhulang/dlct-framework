from keras.models import load_model
import keras.backend as K
import keras.losses
from keras.regularizers import l2

import os
import glob
import time
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

from models import *
from data_load import *
from utils.file_load_save import *
from utils.metrics import *
from utils.plot_function import *
import helper_functions as hf
from utils.loss_fns import *

def parse_mf(model_folder, dvkey = 'v-', crosskey = 'cross-'):
	import re
	import os
	x = re.search(dvkey+'\d+', model_folder)
	dat_ver = x.group(0).split('-')[-1]
	dat_ver_int = int(float(dat_ver))
	y = re.search(crosskey+'\d+',model_folder)
	cross_nb = y.group(0).split('-')[-1]
	cross_nb_int = int(float(cross_nb))
	result_folder_str = os.path.split(model_folder[:x.end()])[1]
	return dat_ver_int, cross_nb_int, result_folder_str

def model_evaluate(model_folder, results_root_folder, fnc_str = 'FCN_A'):
	fcns = globals()['{}'.format(fnc_str)]
	# parse the model folder
	dat_ver, cross_nb, result_folder = parse_mf(model_folder)
	## load the model and prediction
	# load model
	model_file = os.path.join(model_folder, 'model.h5')
# 	print(model_file)
	model = load_model(model_file)
	# save weight file
	weight_file = os.path.join(model_folder, 'weights.h5')
	model.save_weights(weight_file)
	# load weights to model with larger input size
# 	model = fcns((128,128))
# 	model = fcns((600,600))
	model = fcns((256,256))
# 	model = fcns((1200,1200))
	model.load_weights(weight_file, by_name = True)
	# load data with data version and cross validation number
	X_train,X_val,Y_train,Y_val = load_train_data(val_version = dat_ver, cross_val = cross_nb)
	if len(X_val.shape)<3:
		X_val = X_val.reshape(X_val.shape+(1,))
	time1 = time.time()
	preds = model.predict(X_val)/100
	time2 = time.time()
	shp = preds.shape
	preds = np.squeeze(preds)
	pred_counts = np.apply_over_axes(np.sum,preds,[1,2]).reshape(preds.shape[0])
	ground_counts = np.apply_over_axes(np.sum,Y_val,[1,2]).reshape(preds.shape[0])
	mean_abs_error = np.mean(np.abs(pred_counts-ground_counts))
	std_abs_error = np.std(np.abs(pred_counts-ground_counts))
	# 		print(np.mean(pred_counts),np.mean(ground_counts))
	# 		print(mean_abs_error,std_abs_error)
	acc = (np.mean(ground_counts)-mean_abs_error)/np.mean(ground_counts)
	print(model_file)
	print('prediction time:{0:.2f}, acc:{1:0.3f}, mae:{2:0.3f}, std:{3:0.3f}'.format(time2-time1, acc, mean_abs_error,std_abs_error))
# 	_, _cosine = density_Cosine_calculate_for_array(preds,Y_val.squeeze())
# 	print('prediction time:{0:.2f}, acc:{1:0.3f}, cosine:{2:0.3f}'.format(time2-time1, acc, _cosine))
	
	# save the results
# 	full_result_folder = os.path.join(results_root_folder, result_folder)
# 	hf.generate_folder(full_result_folder)
# 
# 	for i in range(preds.shape[0]):
# 		file_name = os.path.join(full_result_folder,'{}-{}.pkl'.format(cross_nb,i))
# 		save_pickles(file_name,preds[i].squeeze(), Y_val[i].squeeze(), X_val[i].squeeze(), keys = ['est_den', 'gt_den', 'ori_img'])
	
	K.clear_session()

def model_evaluate_sequence(model_folder, results_root_folder, fnc_str = 'FCN_A'):
	import glob
	import re
	from natsort import natsorted
	fcns = globals()['{}'.format(fnc_str)]
	# parse the model folder
	dat_ver, cross_nb, result_folder = parse_mf(model_folder)
# 	model_file_list = glob.glob(model_folder+'/model*.h5')
	model_file_list = glob.glob(model_folder+'/weights*.h5')
# 	print(model_file_list)
	model_file_list = natsorted(model_file_list)
	acc_list = []
	mae_list = []
	std_list = []
# 	input_shape = (608,608)
	input_shape = (256,256)
	model = fcns(input_shape)
	best_err = float('inf')
	best_indx = 0
	for i, model_file in enumerate(model_file_list):		
		## load the model and prediction
		# load model
# 		model_file = os.path.join(model_folder, 'model.h5')
	# 	print(model_file)
# 		model = load_model(model_file)
	# 	print('----1---')
		# save weight file
# 		weight_file = os.path.join(model_folder, 'weights.h5')
# 		model.save_weights(weight_file)
	# 	print('----2---')
		# load weights to model with larger input size
	# 	model = fcns((128,128))
	# 	model = fcns((600,600))
# 		model = fcns((256,256))
	# 	model = fcns((1200,1200))
# 		model.load_weights(weight_file, by_name = True)
		model.load_weights(model_file, by_name = True)
		# load data with data version and cross validation number
		X_train,X_val,Y_train,Y_val = load_train_data(val_version = dat_ver, cross_val = cross_nb, normal_proc = True)
		shp = X_val.shape
		if shp[1:3] == (600,600):
# 			X_zap = np.zeros((shp[0],)+input_shape+(3,), dtype = np.uint8)
			X_zap = np.zeros((shp[0],)+input_shape+(3,))
			X_zap[:,4:-4,4:-4,:] = X_val
			X_val = X_zap
		if len(X_val.shape)<3:
			X_val = X_val.reshape(X_val.shape+(1,))
		time1 = time.time()
		x_len = X_val.shape[0]
# 		preds = model.predict(X_val[:int(x_len/3),:])/100
# 		preds1 = model.predict(X_val[int(x_len/3):int(x_len*2/3),:])/100
# 		preds2 = model.predict(X_val[int(x_len*2/3):,:])/100
# 		pred_list = []
# 		for k in range(x_len):
# 			pred_list.append(model.predict(X_val[k:k+1,:])/100)
# 		preds = np.concatenate(pred_list, axis = 0)
		preds_list = model.predict(X_val)
		if not type(preds_list) is list:
			preds_list = [preds_list]
		mae_list = []
		for preds in preds_list:
			preds = preds/100
# 			preds = model.predict(X_val)[-1]/100
	# 		preds = np.concatenate([preds,preds1,preds2], axis = 0)
			time2 = time.time()
			shp = preds.shape
			preds = np.squeeze(preds)
			pred_counts = np.apply_over_axes(np.sum,preds,[1,2]).reshape(preds.shape[0])
			ground_counts = np.apply_over_axes(np.sum,Y_val,[1,2]).reshape(preds.shape[0])
			mean_abs_error = np.mean(np.abs(pred_counts-ground_counts))
			mae_list.append(mean_abs_error)
		print(mae_list)
# 		std_abs_error = np.std(np.abs(pred_counts-ground_counts))
# 		# 		print(np.mean(pred_counts),np.mean(ground_counts))
# 		# 		print(mean_abs_error,std_abs_error)
# 		acc = (np.mean(ground_counts)-mean_abs_error)/np.mean(ground_counts)
# # 		print(model_file)
# 		print('prediction time:{0:.2f}, acc:{1:0.3f}, mae:{2:0.3f}, std:{3:0.3f}'.format(time2-time1, acc, mean_abs_error,std_abs_error))
# 		acc_list.append(acc)
# 		mae_list.append(mean_abs_error)
# 		std_list.append(std_abs_error)
# 		if best_err >mean_abs_error:
# 			file_name = os.path.basename(model_file)
# 			re_res = re.search('\d+', file_name)
# 			idx = int(float(re_res.group(0)))
# 			best_idx = idx
# 			best_err = mean_abs_error
# 	_, _cosine = density_Cosine_calculate_for_array(preds,Y_val.squeeze())
# 	print('prediction time:{0:.2f}, acc:{1:0.3f}, cosine:{2:0.3f}'.format(time2-time1, acc, _cosine))
	
	# save the results
# 	full_result_folder = os.path.join(results_root_folder, result_folder)
# 	hf.generate_folder(full_result_folder)
# 
# 	for i in range(preds.shape[0]):
# 		file_name = os.path.join(full_result_folder,'{}-{}.pkl'.format(cross_nb,i))
# 		save_pickles(file_name,preds[i].squeeze(), Y_val[i].squeeze(), X_val[i].squeeze(), keys = ['est_den', 'gt_den', 'ori_img'])
# 	pickle_file_name = os.path.join(model_folder, 'eval_result.pkl')
# 	save_result_pkl(pickle_file_name,acc_list, mae_list, std_list)
# 	plot_metrics(model_folder, 'eval_result.png', mae_list,std_list)
# 	min_idx = np.argmin(mae_list)
# 	print('best epoch idx: {0}, mae:{1:0.3f}-std:{2:0.3f}'.format(best_idx, mae_list[min_idx],std_list[min_idx]))
# 	best_model_file_name = os.path.join(model_folder, 'best_weights_{}.h5'.format(best_idx))
# 	print('{}-{}'.format(os.path.basename(best_model_file_name), os.path.basename(model_file_list[min_idx])))
# 	model.load_weights(model_file_list[min_idx], by_name = True)
# 	model.save_weights(best_model_file_name)
	K.clear_session()

def save_result_pkl(file_name,acc_list, mae_list, std_list, keys = ['acc', 'mae', 'std']):
	save_pickles(file_name,acc_list, mae_list, std_list, keys = ['acc', 'mae', 'std'])
# 	pickle_item = {keys[0]:acc_list,keys[1]:mae_list,keys[2]:std_list}
# 	pickle.dump(pickle_item, open(file_name, 'wb'))

def read_result_pkl(file_name,acc_list, mae_list, std_list, keys = ['acc', 'mae', 'std']):
	return read_pickles(file_name, keys = ['acc', 'mae', 'std'])

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "4"
	keras.losses.mse_err = mse_err
	keras.losses.mse_ct_err = mse_ct_err
	## the parameters
	models_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/models')
	results_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/results')
# 	fnc_str = 'FCRN_A'
# 	fnc_str = 'URN'
# 	fnc_str = 'U_Net_FCRN_A'
# 	fnc_str = 'MCNN'
# 	fnc_str = 'MCNN_U'
# 	fnc_str = 'imp_MCNN_U2'
# 	fnc_str = 'buildModel_FCRN_A'
	fnc_str = 'buildMultiModel_FCRN_A'
# 	fnc_str = 'buildMultiModel_InCep'
# 	fnc_str = 'buildModel_InCep'
# 	fnc_str = 'buildModel_InCep_v2'
# 	fnc_str = 'buildModel_MCNN_U'
# 	fnc_str = 'MCNN_U_x4'
# 	loss_fn = 'mse_ct_err'
# 	loss_fn = 'mse_err'
	loss_fn = 'mse'

# 	date = '4.29'
# 	date = '4.30'
# 	date = '5.3'
# 	date = '5.4'
# 	date = '5.9'
# 	date = '5.6'
	date = '5.17'
# 	date = '5.6'
	data_version = 24
# 	data_version = 25
# 	model_folder = os.path.join(models_root_folder,'date-4.27-FCRN_A-mse_err-v-24-cross-0-batch-320-lr-0.001-r-0')
# 	model_evaluate(model_folder, results_root_folder, fnc_str = fnc_str)
# 	model_folders_ptrn = 'date-{}-{}-mse*v-{}*'.format(date,fnc_str,data_version)
	model_folders_ptrn = 'date-{}-{}-{}*v-{}*norm*'.format(date,fnc_str,loss_fn,data_version)
	final_model_folders_ptrn = os.path.join(models_root_folder, model_folders_ptrn)
	model_folders = glob.glob(final_model_folders_ptrn)
	for i in range(len(model_folders)):
# 		print(model_folders[i])
		model_folder = model_folders[i]
		print(model_folder)
# 		model_evaluate(model_folder, results_root_folder, fnc_str = fnc_str)
		model_evaluate_sequence(model_folder, results_root_folder, fnc_str = fnc_str)
# 	model_folders_ptrn = 'date-{}-{}-{}-v-{}-cross-{}-batch-{}-lr-{}-r-{}'.format(date,net_arch,loss_fn,val_version,0, batch_size,lr,ratio)
# 	model_folders_ptrn = 'date-{}-{}-v-{}*'.format(date,fnc_str,data_version)
# 	final_model_folders_ptrn = os.path.join(models_root_folder, model_folders_ptrn)
# 	model_folders = glob.glob(final_model_folders_ptrn)
# 	for i in range(len(model_folders)):
# 		print(model_folders[i])
# 		model_evaluate(model_folders[i], results_root_folder, fnc_str = fnc_str)
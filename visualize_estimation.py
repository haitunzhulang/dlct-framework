import glob
import matplotlib.pyplot as plt
import os
import numpy as np

from utils.file_load_save import *

result_folder = os.path.expanduser('~/dl-cells/dlct-framework/results')
model_folder = 'date-4.29-FCRN_A-mse_ct_err-v-24'
estimte_folder = os.path.join(result_folder, model_folder, '*.pkl')
file_names = glob.glob(estimte_folder)
est_list = []
den_list = []
img_list = []
for i, f in enumerate(file_names):
	file_name = file_names[i]
	est, den, img = read_pickles(file_name, keys = ['est_den', 'gt_den', 'ori_img'])
	est_list.append(est)
	den_list.append(den)
	img_list.append(img)
	
fig = plt.figure()
plt.ion()
fig.clf()
for i in range(len(est_list)):
	ax = fig.add_subplot(1,3,1)
	bx = fig.add_subplot(1,3,2)
	cx = fig.add_subplot(1,3,3)
	ax.imshow(np.squeeze(img_list[i]))
	bx.imshow(np.squeeze(den_list[i]))
	cx.imshow(np.squeeze(est_list[i]))
	bx.set_xlabel('cell count:{0:0.3f}'.format(np.sum(den_list[i])))
	cx.set_xlabel('cell count:{0:0.3f}'.format(np.sum(est_list[i])))
	plt.pause(0.5)


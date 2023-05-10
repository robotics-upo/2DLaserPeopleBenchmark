import math
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import auc
np.tau = math.tau

DATASET_FILE = 'frog_16-41_test.h5'

#RESULTS_FILE = 'petra_frog.npz'
#RESULTS_FILE = 'petra_frog_mixedloss.npz'
#RESULTS_FILE = 'LFE_seg_20230421145111.npz'
#RESULTS_FILE = 'LFE_PPN_20230427145355.npz'
#RESULTS_FILE = 'drow_on_frog.npz'
#RESULTS_FILE = 'dr_spaam_1_on_frog.npz'
RESULTS_FILE = 'dr_spaam_5_on_frog.npz'

ASSOC_DISTANCE = 0.5 #0.3

with h5py.File(DATASET_FILE, 'r') as f:
	gt_people      = f['circles'][:,0:2]
	gt_people_idxs = f['circle_idx'][:]
	gt_people_nums = f['circle_num'][:]

with np.load('test/'+RESULTS_FILE) as f:
	det_people = f['people']
	det_people = det_people.reshape((-1,3)).astype(np.float32)
	det_people_idxs = f['idxs']
	det_people_nums = f['nums']

NUM_SCANS = gt_people_nums.shape[0]
assert det_people_nums.shape[0] == NUM_SCANS

def scan_gt_slice(i):
	return slice(gt_people_idxs[i], gt_people_idxs[i] + gt_people_nums[i])

def scan_det_slice(i):
	return slice(det_people_idxs[i], det_people_idxs[i] + det_people_nums[i])

def count_TPs(gt, det, max_distance):
	distances = det[:,None,:] - gt[None,:,:]
	distances = np.hypot(distances[:,:,0], distances[:,:,1])

	det_gt_match = np.argmin(distances, axis=1)
	det_gt_dist  = np.min(distances, axis=1)

	TP = 0
	matched_gt = np.zeros((gt.shape[0],), dtype=np.uint8)
	for i_det in np.argsort(det_gt_dist, kind='stable'):
		if det_gt_dist[i_det] > max_distance: break
		i_gt = det_gt_match[i_det]
		if matched_gt[i_gt]: continue
		matched_gt[i_gt] = 1
		TP += 1

	return TP

def matchup(gt, det, max_distance=ASSOC_DISTANCE):
	out = []

	# Assuming det is in descending score order
	last_cumTP = 0
	for i in range(det.shape[0]):
		cumTP = count_TPs(gt, det[:i+1,1:3], max_distance)
		TP = cumTP - last_cumTP
		last_cumTP = cumTP
		assert TP >= 0
		out.append(TP)

	return np.array(out, dtype=np.uint8)

det_TP = np.empty((det_people.shape[0],), dtype=np.uint8)

for i in tqdm(range(gt_people_idxs.shape[0])):
	cur_gt = gt_people[gt_people_idxs[i]:gt_people_idxs[i]+gt_people_nums[i]]
	cur_det = det_people[det_people_idxs[i]:det_people_idxs[i]+det_people_nums[i]]
	det_TP[det_people_idxs[i]:det_people_idxs[i]+det_people_nums[i]] = matchup(cur_gt, cur_det)

Pr_values = [ ]
Rc_values = [ ]
Th_values = [ ]

num_dets = det_people.shape[0]
det_order = np.argsort(-det_people[:,0], axis=0, kind='stable')
det_is_boundary = det_people[det_order[1:],0] < det_people[det_order[0:num_dets-1],0]

cum_TP = 0
for i,idx in enumerate(det_order):
	cum_TP += det_TP[idx]
	if i==num_dets-1 or det_is_boundary[i]:
		Pr_values.append(cum_TP / float(i+1))
		Rc_values.append(cum_TP / float(gt_people.shape[0]))
		Th_values.append(idx)

Pr_values = np.array(Pr_values, dtype=np.float32)
Rc_values = np.array(Rc_values, dtype=np.float32)
Th_values = det_people[np.array(Th_values,dtype=np.uint32),0]

print("Saving results...")
np.savez_compressed('test/'+RESULTS_FILE.replace('.npz', f'_pr_{ASSOC_DISTANCE}.npz'),
	R=Rc_values,
	P=Pr_values,
	T=Th_values,
)

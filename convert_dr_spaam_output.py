import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#DIR_NAME = 'drow_on_frog'
#DIR_NAME = 'dr_spaam_1_on_frog'
DIR_NAME = 'dr_spaam_5_on_frog'
NUM_SCANS = 50088

SCORE_THRESHOLD = 0.01
SCAN_FAR = 10.0

all_people = []
all_people_idx = []
all_people_num = []

total_people_added = 0

for i in tqdm(range(NUM_SCANS)):
	with open(f'test/{DIR_NAME}/{i:06d}.txt', 'r', encoding='utf-8') as f:
		lines = f.readlines()

	cur_people = []
	for l in lines:
		l = l.strip(' \t\r\n')
		if l == '': continue
		l = l.split(' ')
		c_score = float(l[-1])
		c_x     = float(l[-5])
		c_y     = float(l[-4])
		c_dist  = np.hypot(c_x, c_y)
		if c_score < SCORE_THRESHOLD or c_dist > SCAN_FAR: continue

		cur_people.append((c_score, c_x, c_y))

	all_people_idx.append(total_people_added)
	all_people_num.append(len(cur_people))
	total_people_added += len(cur_people)

	if len(cur_people)==0:
		continue

	cur_people = np.array(cur_people, dtype=np.float32)
	cur_people = cur_people[np.argsort(-cur_people[:,0],kind='stable'),:]
	all_people.append(cur_people)

all_people = np.concatenate(all_people, axis=0)
all_people_idx = np.array(all_people_idx, dtype=np.uint32)
all_people_num = np.array(all_people_num, dtype=np.uint32)

print("Saving results...")
np.savez_compressed(f'test/{DIR_NAME}.npz',
	people=all_people,
	idxs=all_people_idx,
	nums=all_people_num)

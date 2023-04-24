import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.data_loader import DataLoader
from utils.models import extract_seg_out
from utils.segmentation import parse_seg, join_dets
from utils.testing import batch_and_infer

DATASET_FILE = 'frog_test.h5'
#MODEL_FILE = 'LFE_20230417143739.h5'
#MODEL_FILE = 'LFE_20230417161234.h5'
#MODEL_FILE = 'LFE_20230420114910-logistic.h5'
#MODEL_FILE = 'LFE_mixed_20230420131937.h5'
#MODEL_FILE = 'LFE_mixed_global_20230421094401.h5'
MODEL_FILE = 'LFE_mixed_global_20230421145111.h5'

mdl = tf.keras.models.load_model(MODEL_FILE, compile=False)

# Create data loader
loader = DataLoader(DATASET_FILE, min_people=0, points_per_sector=0)

print(len(loader),'scans in total')

all_total = 0
all_people_idx = []
all_people_num = []
all_people = []

for i, scan, det_y in batch_and_infer(loader, mdl, pred_xform=extract_seg_out):
	all_people_idx.append(all_total)

	det_peaks, _, det_counts = parse_seg(loader, scan, det_y)
	if det_peaks is None:
		all_people_num.append(0)
		continue

	det_people = join_dets(det_peaks, det_counts, 1.5*loader.HARDCODED_PERSON_RADIUS)
	all_people.append(det_people)
	all_people_num.append(len(det_people))
	all_total += len(det_people)

all_people = np.concatenate(all_people, axis=0)
all_people_idx = np.array(all_people_idx, dtype=np.uint32)
all_people_num = np.array(all_people_num, dtype=np.uint32)

print("Saving results...")
np.savez_compressed(MODEL_FILE.replace('.h5', '_test_out.npz'),
	people=all_people,
	idxs=all_people_idx,
	nums=all_people_num)

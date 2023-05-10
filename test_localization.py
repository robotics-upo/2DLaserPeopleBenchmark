import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.data_loader import DataLoader
from utils.models import extract_loc_out
from utils.localization import parse_loc
from utils.testing import batch_and_infer

DATASET_FILE = 'frog_16-41_test.h5'
MODEL_FILE = 'LFE_PPN_20230427145355.h5'

mdl = tf.keras.models.load_model(MODEL_FILE, compile=False)

# Create data loader
loader = DataLoader(DATASET_FILE, min_people=0, points_per_sector=6)

print(len(loader),'scans in total')

all_total = 0
all_people_idx = []
all_people_num = []
all_people = []

for i, scan, det_y in batch_and_infer(loader, mdl, pred_xform=extract_loc_out):
	all_people_idx.append(all_total)

	det_people = parse_loc(loader, det_y)
	if det_people is None:
		all_people_num.append(0)
		continue

	all_people.append(det_people)
	all_people_num.append(len(det_people))
	all_total += len(det_people)

all_people = np.concatenate(all_people, axis=0)
all_people_idx = np.array(all_people_idx, dtype=np.uint32)
all_people_num = np.array(all_people_num, dtype=np.uint32)

print("Saving results...")
np.savez_compressed('test/'+MODEL_FILE.replace('.h5', '.npz'),
	people=all_people,
	idxs=all_people_idx,
	nums=all_people_num)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .data_loader import DataLoader

def batch_and_infer(loader:DataLoader, mdl:tf.keras.Model, batch_size=128, pred_xform=None):
	total_len = len(loader)
	for base_i in tqdm(range(0, total_len, batch_size)):
		remaining = total_len - base_i
		cur_len = remaining if remaining < batch_size else batch_size
		batch_scan = []
		batch_x = []
		for j in range(cur_len):
			j_scan, _ = loader[base_i+j]
			j_x = loader.normalize_scan(j_scan)
			batch_scan.append(j_scan)
			batch_x.append(j_x)

		batch_x = np.stack(batch_x, axis=0)
		batch_pred = mdl(batch_x[:,:,None])
		if pred_xform is not None: batch_pred = pred_xform(batch_pred)

		for j in range(cur_len):
			yield (base_i+j, batch_scan[j], batch_pred[j])

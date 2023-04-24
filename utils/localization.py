import tensorflow as tf
import numpy as np

from .data_loader import DataLoader

def circle_overlap(circles1, circles2, radius):
	distances = circles1[:,None,:] - circles2[None,:,:]
	distances = np.hypot(distances[:,:,0], distances[:,:,1])
	return np.maximum(0.0, 1.0 - distances / (2*radius))

class LocDataGenerator:
	def __init__(self, loader:DataLoader, split=0, overlap_threshold=0.25):
		self.loader = loader
		self.scanlist = loader.get_split(split)
		self.rng : np.random.Generator
		self.rng = np.random.default_rng()
		self.overlap_threshold = overlap_threshold

	def __len__(self):
		return len(self.scanlist)

	def __call__(self):
		scanlist = np.copy(self.scanlist)
		self.rng.shuffle(scanlist)

		for i in scanlist:
			scan, grid_gt = LocDataGenerator.generate(self.loader, i, self.overlap_threshold)
			yield (scan[:,None], grid_gt)

	@staticmethod
	def generate(loader:DataLoader, i, overlap_threshold):
		scan, gt_circles = loader[i]

		anchor_circle_centers_xy = loader.grid_xy.reshape((-1,2))
		anchor_circle_centers_polar = loader.grid_polar.reshape((-1,2))

		gt_circle_centers_xy = gt_circles[:,0:2]
		gt_circle_centers_polar = gt_circles[:,3:5]

		overlaps = circle_overlap(anchor_circle_centers_xy, gt_circle_centers_xy, loader.HARDCODED_PERSON_RADIUS)

		max_overlap_per_anchor = np.max(overlaps, axis=1)
		gt_circle_assignments = np.argmax(overlaps, axis=1)
		max_overlap_per_gt_circle = np.max(overlaps, axis=0)
		highest_overlap_anchor_idxs,temp = np.where(overlaps==max_overlap_per_gt_circle)
		highest_overlap_anchor_idxs = highest_overlap_anchor_idxs[temp]

		objectness_score = np.full(loader.NUM_SECTORS * loader.NUM_ANCHORS_PER_SECTOR, -1.0, dtype=np.float32)
		objectness_score[max_overlap_per_anchor < overlap_threshold] = 0.0
		objectness_score[max_overlap_per_anchor >= overlap_threshold] = 1.0
		objectness_score[highest_overlap_anchor_idxs] = 1.0

		regr_target = gt_circle_centers_polar[gt_circle_assignments,:] - anchor_circle_centers_polar
		# Convert angles to arcs
		regr_target[:,1] *= anchor_circle_centers_polar[:,0]

		grid_gt = np.empty((loader.NUM_SECTORS * loader.NUM_ANCHORS_PER_SECTOR, 3))
		grid_gt[:,0]   = objectness_score
		grid_gt[:,1:3] = loader.ANCHOR_REGRESSION_SCALE*grid_gt[:,0,None]*regr_target
		grid_gt = grid_gt.reshape((loader.NUM_SECTORS, loader.NUM_ANCHORS_PER_SECTOR, 3))

		return (loader.normalize_scan(scan), grid_gt)

	@staticmethod
	def from_loader(loader:DataLoader, **kwargs) -> tf.data.Dataset:
		return tf.data.Dataset.from_generator(
			g := LocDataGenerator(loader, **kwargs),
			output_signature=(
				tf.TensorSpec(shape=(loader.SCAN_WIDTH, 1), dtype=tf.float32),
				tf.TensorSpec(shape=(loader.NUM_SECTORS, loader.NUM_ANCHORS_PER_SECTOR, 3), dtype=tf.float32),
			),
		).apply(tf.data.experimental.assert_cardinality(len(g)))

def parse_loc(loader:DataLoader, locdata, threshold=0.01):
	a_objness = locdata[:,:,0]
	a_regr    = locdata[:,:,1:3] / loader.ANCHOR_REGRESSION_SCALE
	a_regr[:,:,1] /= loader.grid_polar[:,:,0]
	a_regr_polar = loader.grid_polar + a_regr
	a_regr_xy = np.stack([
		a_regr_polar[:,:,0] * np.cos(a_regr_polar[:,:,1]),
		a_regr_polar[:,:,0] * np.sin(a_regr_polar[:,:,1])
	], axis=2)

	a_objness = np.reshape(a_objness, (-1,))
	a_regr_xy = np.reshape(a_regr_xy, (-1,2))
	a_list    = np.argsort(-a_objness, axis=None, kind='stable')

	a_objness = a_objness[a_list]
	a_regr_xy = a_regr_xy[a_list]

	people_nms = []
	for score, (curx, cury) in zip(a_objness, a_regr_xy):
		if score < threshold: break
		mindist = loader.SCAN_FAR
		for (_,ox,oy) in people_nms:
			mindist = min(mindist, np.hypot(curx-ox,cury-oy))
		if mindist < 2*loader.HARDCODED_PERSON_RADIUS: continue
		people_nms.append((score,curx,cury))

	if len(people_nms)>0:
		return np.array(people_nms, dtype=np.float32)
	else:
		return None

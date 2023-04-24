import tensorflow as tf
import numpy as np
from scipy.signal import find_peaks

from .data_loader import DataLoader

class SegDataGenerator:
	def __init__(self, loader:DataLoader, split=0):
		self.loader = loader
		self.scanlist = loader.get_split(split)
		self.rng : np.random.Generator
		self.rng = np.random.default_rng()

	def __len__(self):
		return len(self.scanlist)

	def __call__(self):
		scanlist = np.copy(self.scanlist)
		self.rng.shuffle(scanlist)

		for i in self.scanlist:
			scan, seg_gt = SegDataGenerator.generate(self.loader, i)
			yield (scan[:,None], seg_gt[:,None])

	@staticmethod
	def generate(loader:DataLoader, i):
		scan, gt_circles = loader[i]
		scan_xy = scan[:,None] * loader.SCAN_POINTS

		gt_people_xy = gt_circles[:,0:2] # X Y
		gt_people_r  = gt_circles[:,2]   # Radius
		gt_people_d  = gt_circles[:,3]   # Distance from origin
		removedpeople = np.nonzero(gt_people_d > loader.SCAN_FAR)[0]

		point_dist = scan_xy[:,None,:] - gt_people_xy[None,:,:]
		point_dist = np.hypot(point_dist[:,:,0], point_dist[:,:,1])
		point_dist[:,removedpeople] = 100.0 # mask these out with an arbitrarily large distance

		return (loader.normalize_scan(scan), np.any(point_dist <= gt_people_r[None,:], axis=1).astype(np.float32))

	@staticmethod
	def from_loader(loader:DataLoader, **kwargs) -> tf.data.Dataset:
		return tf.data.Dataset.from_generator(
			g := SegDataGenerator(loader, **kwargs),
			output_signature=(
				tf.TensorSpec(shape=(loader.SCAN_WIDTH, 1), dtype=tf.float32),
				tf.TensorSpec(shape=(loader.SCAN_WIDTH, 1), dtype=tf.float32),
			),
		).apply(tf.data.experimental.assert_cardinality(len(g)))

def parse_seg(loader:DataLoader, scan, seg):
	peaks, props = find_peaks(seg, height=0.01, prominence=0.1, width=1, rel_height=0.5)
	if len(peaks)==0: return (None, [], [])

	scan_xy = scan[:,None] * loader.SCAN_POINTS

	left_ips = props['left_ips']
	right_ips = props['right_ips']
	peak_heights = props['peak_heights']

	scores = []
	centers = []
	counts = []

	for i,peak in enumerate(peaks):
		left_idx = int(np.ceil(left_ips[i]))
		right_idx = int(np.floor(right_ips[i]))
		peak_point = scan_xy[peak,:]
		scores.append(peak_heights[i])
		counts.append(right_idx - left_idx + 1)
		if left_idx==right_idx:
			centers.append(peak_point)
			continue

		# Find the closest point
		closest_point_idx = np.argmin(scan[left_idx:right_idx]) + left_idx
		closest_point = scan_xy[closest_point_idx,:]

		# Check if the peak is far from the closest point (it could be a point in the middle of two legs)
		pdist = peak_point - closest_point
		pdist = np.hypot(pdist[0], pdist[1])
		if pdist <= loader.HARDCODED_PERSON_RADIUS:
			centers.append(peak_point)
			continue

		# Otherwise: compute a new center close to the closest point
		all_points = scan_xy[left_idx:right_idx,:]
		pdist = all_points[:,None,:] - closest_point[None,None,:]
		pdist = np.hypot(pdist[:,0,0], pdist[:,0,1])
		chosen_points = all_points[np.nonzero(pdist <= 2*loader.HARDCODED_PERSON_RADIUS)[0],:]
		centers.append(np.mean(chosen_points, axis=0))

	scores = np.stack(scores, axis=0)
	centers = np.stack(centers, axis=0)
	return (np.concatenate([ scores[:,None], centers ], axis=1), peaks, counts)

def join_dets(dets, counts, radius):
	out = []
	cnt = []

	def find_close_point(pt):
		for i,ref in enumerate(out):
			d = pt[1:3] - ref[1:3]
			d = np.hypot(d[0], d[1])
			if d <= radius: return i
		return None

	# This is basically NMS, but merging instead of discarding
	for i in np.argsort(-dets[:,0], kind='stable'):
		cur_det = dets[i,:]
		cur_cnt = counts[i]

		j = find_close_point(cur_det)
		if j is not None:
			# Merge with detection
			other_det = out[j]
			other_cnt = cnt[j]
			cnt[j] += cur_cnt
			alpha = float(cur_cnt) / float(cur_cnt + other_cnt)
			other_det[:] = alpha * cur_det + (1.-alpha) * other_det

			# Ensure detection array stays sorted
			out.sort(key=lambda r: r[0], reverse=True)
		else:
			# New point, add it to the list
			out.append(cur_det)
			cnt.append(cur_cnt)

	return np.stack(out, axis=0)

import numpy as np
import h5py

class DataLoader:
	def __init__(self, filename, min_people=0, points_per_sector=0):
		f = h5py.File(filename, 'r')
		self.f           = f # keep it open
		self.scans       = f['scans'] # lazy load
		self.timestamps  = f['timestamps'][:]
		self.circle_idxs = f['circle_idx'][:]
		self.circle_nums = f['circle_num'][:]
		self.circles     = f['circles'] # lazy load
		split            = f.get('split') # optional
		if split: self.split = split[:]
		else: self.split = None

		self.SCAN_WIDTH = self.scans.shape[1]
		if self.SCAN_WIDTH == 450:
			# DROW
			self.SCAN_NEAR   = 0.2
			self.SCAN_FAR    = 15.0
			self.SCAN_FOV    = np.radians(225)
			self.HARDCODED_PERSON_RADIUS = 0.35
		elif self.SCAN_WIDTH == 720:
			# FROG
			self.SCAN_NEAR   = 0.2
			self.SCAN_FAR    = 10.0
			self.SCAN_FOV    = np.radians(180)
			self.HARDCODED_PERSON_RADIUS = 0.4
		else:
			raise Exception("Unknown dataset")

		self.selection = np.nonzero(self.circle_nums >= min_people)[0]

		self.SCAN_ANGMIN = - self.SCAN_FOV/2
		self.SCAN_ANGMAX = + self.SCAN_FOV/2
		self.SCAN_ANGLES = np.linspace(self.SCAN_ANGMIN, self.SCAN_ANGMAX, self.SCAN_WIDTH, endpoint=False, dtype=np.float32)
		self.SCAN_POINTS = np.stack([ np.cos(self.SCAN_ANGLES), np.sin(self.SCAN_ANGLES) ], axis=-1)

		if points_per_sector > 0:
			self.NUM_SECTORS = self.SCAN_WIDTH // points_per_sector
			self.NUM_ANCHORS_PER_SECTOR = int((self.SCAN_FAR - self.SCAN_NEAR) / (0.8*self.HARDCODED_PERSON_RADIUS))

			self.SECTOR_AMPL = self.SCAN_FOV / self.NUM_SECTORS
			self.SECTOR_ANGLES = np.linspace(self.SCAN_ANGMIN + self.SECTOR_AMPL/2, self.SCAN_ANGMAX - self.SECTOR_AMPL/2, self.NUM_SECTORS, endpoint=True)

			self.ANCHOR_DEPTH = (self.SCAN_FAR - self.SCAN_NEAR) / self.NUM_ANCHORS_PER_SECTOR
			self.ANCHOR_DISTANCES = np.linspace(self.SCAN_NEAR + self.ANCHOR_DEPTH/2, self.SCAN_FAR - self.ANCHOR_DEPTH/2, self.NUM_ANCHORS_PER_SECTOR, endpoint=True)

			self.ANCHOR_REGRESSION_SCALE = 1.0/self.ANCHOR_DEPTH

			g_dist,g_ang = np.meshgrid(self.ANCHOR_DISTANCES, self.SECTOR_ANGLES)
			g_x = g_dist * np.cos(g_ang)
			g_y = g_dist * np.sin(g_ang)

			self.grid_polar = np.stack((g_dist, g_ang), axis=2)
			self.grid_xy    = np.stack((g_x, g_y), axis=2)

	def get_split(self, i):
		if self.split is not None:
			return np.nonzero(self.split[self.selection]==i)[0]
		else:
			return np.array(range(len(self.selection)))

	def normalize_scan(self, scan):
		return 1. - np.clip(scan, self.SCAN_NEAR, self.SCAN_FAR) / self.SCAN_FAR

	def __len__(self):
		return self.selection.shape[0]

	def __getitem__(self, i):
		i = self.selection[i]
		return (np.nan_to_num(self.scans[i,:], posinf=100.0), self.circles[self.circle_idxs[i]:self.circle_idxs[i]+self.circle_nums[i]])

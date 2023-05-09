import numpy as np
import glob
import json
from typing import Iterator

FROG_PATH = '/mnt/data/work/datasets/frog_all'

BAGMAP = {
	'10-31': '2014-04-30-10-31-26',
	'11-36': '2014-04-29-11-36-22', # trainval1
	'12-43': '2014-04-30-12-43-38', # trainval2
	'14-57': '2014-04-29-14-57-50',
	'15-53': '2014-04-28-15-53-18',
	'16-41': '2014-04-29-16-41-49', # test
}

SCAN_NEAR   = 0.2
SCAN_FAR    = 10.0
SCAN_FOV    = np.radians(180)

SCAN_ANGMIN = - SCAN_FOV/2
SCAN_ANGMAX = + SCAN_FOV/2

def read_file_line_by_line(filename) -> Iterator[str]:
	with open(filename, 'r', encoding='utf-8') as f:
		for line in f:
			yield line.rstrip('\r\n')

def parse_circles(out, circles):
	temp = []
	for cir in circles:
		if cir.get('type', 1) != 1:
			continue # Ignore non-people
		x = cir['x']
		y = cir['y']
		r = cir['r']
		dist  = np.hypot(x, y)
		angle = np.arctan2(y, x)
		angr  = np.arctan2(r, dist)
		if dist < SCAN_NEAR or dist >= SCAN_FAR or angle < SCAN_ANGMIN or angle >= SCAN_ANGMAX:
			continue # Ignore out of bounds people
		temp.append(np.array([x,y,r,dist,angle,angr], np.float32))
	temp.sort(key=lambda c: (-c[3], c[4])) # Sort back-to-front and CCW
	out.extend(temp)

for bag_id in BAGMAP.keys():
	print(bag_id)

	bag_seq = []
	bag_circles = []
	bag_circle_idx = []
	bag_circle_num = []
	for csv_file in glob.glob(f'{FROG_PATH}/circles/bag_{bag_id}_*.csv'):
		for i,line in enumerate(read_file_line_by_line(csv_file)):
			if i==0: continue
			line = line.split(',', maxsplit=2)
			bag_seq.append(int(line[0]))
			bag_circle_idx.append(cstart := len(bag_circles))
			parse_circles(bag_circles, json.loads(line[2].replace("'", '"')))
			bag_circle_num.append(len(bag_circles) - cstart)

	bag_num_scans = max(bag_seq)+1
	bag_circle_idx_sorted = np.zeros((bag_num_scans,), np.uint32)
	bag_circle_num_sorted = np.zeros((bag_num_scans,), np.uint32)
	print('  ',len(bag_circle_idx),'/',bag_num_scans,'scans annotated')

	for i,seq in enumerate(bag_seq):
		bag_circle_idx_sorted[seq] = bag_circle_idx[i]
		bag_circle_num_sorted[seq] = bag_circle_num[i]

	np.savez_compressed(f'{FROG_PATH}/interm/frog_{bag_id}_circles.npz',
		circles    = bag_circles,
		circle_idx = bag_circle_idx_sorted,
		circle_num = bag_circle_num_sorted,
	)

	print('  ',len(bag_circles),'people exported')

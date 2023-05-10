import h5py
import numpy as np

FROG_PATH = '/mnt/data/work/datasets/frog_all'

BAGMAP = {
	'10-31': '2014-04-30-10-31-26',
	'11-36': '2014-04-29-11-36-22', # trainval1
	'12-43': '2014-04-30-12-43-38', # trainval2
	'14-57': '2014-04-29-14-57-50',
	'15-53': '2014-04-28-15-53-18',
	'16-41': '2014-04-29-16-41-49', # test
}

def define_file(*args, **kwargs):
	kwargs['list'] = args
	return kwargs

FILE_LIST = [
	define_file('11-36', '12-43', suffix='train_val', strip_empty=True, split_bound=0.1),
	define_file('16-41',          suffix='test',      strip_empty=True),
	define_file('10-31'),
	define_file('14-57'),
	define_file('15-53'),
]

def load_one_bag(name, strip_empty=False, **kwargs):
	with np.load(f'{FROG_PATH}/interm/frog_{name}_circles.npz') as f:
		circles = f['circles']
		circle_idx = f['circle_idx']
		circle_num = f['circle_num']

	with np.load(f'{FROG_PATH}/interm/frog_{name}_scans.npz') as f:
		scans = f['data']
		scan_ts = f['ts']

	if strip_empty:
		idx        = np.nonzero(circle_num)[0]
		print('    ', len(scans), '->', len(idx))
		scan_ts    = scan_ts[idx]
		scans      = scans[idx]
		circle_idx = circle_idx[idx]
		circle_num = circle_num[idx]

	return scan_ts, scans, circles, circle_idx, circle_num

def merge_bags(bags, **kwargs):
	if len(bags) == 1:
		return bags[0]

	# Indices need special processing first
	counter = 0
	for i in range(1, len(bags)):
		counter += len(bags[i-1][2]) # add number of circles of previous bag
		bags[i][3][i] += counter     # add offset to circle indices of current bag

	return tuple(np.concatenate([ x[i] for x in bags ], axis=0) for i in range(len(bags[0])))

def load_many_bags(bag_list, **kwargs):
	ret = []
	for bag_name in bag_list:
		print('  ', bag_name)
		ret.append(load_one_bag(bag_name, **kwargs))
	return merge_bags(ret, **kwargs)

for cfg in FILE_LIST:
	cur_list = cfg['list']
	cur_name = '_'.join(cur_list)
	if suffix := cfg.get('suffix', None):
		cur_name += f'_{suffix}'

	print(cur_name)

	cur_scan_ts, cur_scans, cur_circles, cur_circle_idx, cur_circle_num = load_many_bags(cur_list, **cfg)
	cur_split = None

	print('  ','Loaded',cur_scans.shape[0],'scans and',cur_circles.shape[0],'people')
	#continue

	if (bound := cfg.get('split_bound', None)) is not None:
		idx_split = np.random.default_rng().permutation(cur_scans.shape[0])
		split_cutpos = int(.5 + bound*len(idx_split))

		cur_split = np.empty((cur_scans.shape[0],), dtype=np.uint8)
		cur_split[idx_split[:split_cutpos]] = 1 # Validation
		cur_split[idx_split[split_cutpos:]] = 0 # Training

	with h5py.File(f'{FROG_PATH}/frog_{cur_name}.h5', 'w') as f:
		f.create_dataset('timestamps', data=cur_scan_ts)
		f.create_dataset('scans', data=cur_scans)
		f.create_dataset('circles', data=cur_circles)
		f.create_dataset('circle_idx', data=cur_circle_idx)
		f.create_dataset('circle_num', data=cur_circle_num)
		if cur_split is not None:
			f.create_dataset('split', data=cur_split)

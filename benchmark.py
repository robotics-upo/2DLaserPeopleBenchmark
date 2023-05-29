import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import auc
from itertools import cycle
np.tau = math.tau

ASSOC_DISTANCE = 0.5 #0.3

MODELS = {
	'PeTra':          'petra_frog',
	'PeTra*':         'petra_frog_mixedloss',
	'LFE-Peaks':      'LFE_seg_20230421145111',
	'LFE-PPN':        'LFE_PPN_20230427145355',
	'DROW3':          'drow_on_frog',
	'DR-SPAAM (T=1)': 'dr_spaam_1_on_frog',
	'DR-SPAAM (T=5)': 'dr_spaam_5_on_frog',
}

linecycler = cycle(['dotted', 'dashed', 'dashdot'])

fig, ax = plt.subplots(figsize=(6.0, 6.0))

ax.plot([0,1],[0,1], color=(0.8,0.8,0.8), linewidth=0.5, linestyle='dashed')
cm = plt.get_cmap('Set1')
ax.set_prop_cycle(color=cm(np.linspace(0, 1, len(MODELS))))

def proper_ap(recs, precs, points=11):
	new_r = np.linspace(0.0, 1.0, num=points, endpoint=True)
	new_p = []
	for rec_threshold in new_r:
		point_idxs = np.nonzero(recs >= rec_threshold)[0]
		if len(point_idxs) > 0:
			new_p.append(np.max(precs[point_idxs]))
		else:
			new_p.append(0.0)
	new_p = np.array(new_p, dtype=np.float32)
	return np.mean(new_p)

def eer(recs, precs):
	# Find the first nonzero or else (0,0) will be the EER :)
	def first_nonzero_idx(arr):
		return np.nonzero(arr)[0][0]

	p1 = first_nonzero_idx(precs)
	r1 = first_nonzero_idx(recs)
	q1 = max(p1,r1)
	idx = np.argmin(np.abs(precs[q1:] - recs[q1:]))
	return (precs[q1+idx] + recs[q1+idx])/2  # They are often the exact same, but if not, use average.

# Load PR curves and calculate metrics
for name,prfile in MODELS.items():
	with np.load(f'test/{prfile}_pr_{ASSOC_DISTANCE}.npz') as f:
		Pr_values = f['P']
		Rc_values = f['R']
		Th_values = f['T']

	F1_values = 2 * Pr_values * Rc_values / np.clip(Pr_values + Rc_values, 1e-16, 2+1e-16)
	AP = proper_ap(Rc_values, Pr_values)
	i_peakF1 = np.argmax(F1_values)
	peakF1 = F1_values[i_peakF1]
	peakF1_th = Th_values[i_peakF1]
	EER = eer(Rc_values, Pr_values)

	print(f'---- {name} ----')
	print(f'AP:      {AP    *100.0:.1f}')
	print(f'Peak F1: {peakF1*100.0:.1f}')
	print(f'EER:     {EER   *100.0:.1f}')
	print(f'Thresh:  {peakF1_th:.6f}')

	ax.plot(Rc_values, Pr_values, label=name, linewidth=1.5, linestyle=next(linecycler))

SMALL_OFFSET=0.01
ticks = np.linspace(start=0.0, stop=1.0, endpoint=True, num=11)

ax.set_title(f'P-R Curve (d = {ASSOC_DISTANCE})')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(xmin=0.0-SMALL_OFFSET, xmax=1.0+SMALL_OFFSET)
ax.set_xticks(ticks)
ax.set_ylim(ymin=0.0-SMALL_OFFSET, ymax=1.0+SMALL_OFFSET)
ax.set_yticks(ticks)
ax.set_facecolor((0.95,0.95,0.95))
ax.grid(color=(0.8,0.8,0.8))
ax.legend(loc='lower left')

fig.tight_layout(pad=.0)
plt.show()

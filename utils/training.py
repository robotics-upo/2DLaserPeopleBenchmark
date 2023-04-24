import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime

def pretty_train(mdl, name, early_stopping=None, **train_args):
	name = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
	mdl_folder = f'out/{name}'

	if 'callbacks' not in train_args:
		train_args['callbacks'] = []

	if early_stopping is not None:
		train_args['callbacks'].append(early_stopping)

	print(f"Train START: {name}")
	os.makedirs(mdl_folder)

	try:
		mdl.fit(**train_args)
	except KeyboardInterrupt:
		if early_stopping is not None:
			print('Training interrupted, restoring best weights')
			if (best_weights := early_stopping.best_weights) is not None:
				mdl.set_weights(best_weights)

	print('Saving weights and history')

	mdl.save(f"{mdl_folder}/{name}.h5", include_optimizer=False)

	if (history := mdl.history) is not None:
		for metric_name, metric_train in history.history.items():
			if metric_name.startswith('val_'): continue
			metric_val = history.history['val_'+metric_name]
			plt.figure()
			plt.plot(history.epoch, metric_train, color='blue', label='Train')
			plt.plot(history.epoch, metric_val,   color='red',  label='Val', linestyle="--")
			plt.xlabel('Epoch')
			plt.ylabel(metric_name)
			plt.grid(True)
			plt.legend()
			plt.savefig(f"{mdl_folder}/{metric_name}.eps")
			np.savetxt(f"{mdl_folder}/{metric_name}.csv", np.stack([metric_train, metric_val], axis=1), fmt='%5.6f', delimiter=",")

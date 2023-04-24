import tensorflow as tf
from utils.data_loader import DataLoader
from utils.segmentation import SegDataGenerator
from utils.models import build_seg_model, dice_loss, mixed_loss, logistic_adapter
from utils.training import pretty_train

DATASET_FILE = 'frog_train+val.h5'

MODEL_NAME = 'LFE_mixed_global'
NUM_EPOCHS = 100
BATCH_SIZE = 32

# Create data loader
loader = DataLoader(DATASET_FILE, min_people=1, points_per_sector=0)
print(len(loader),'scans in total')

ds_train = SegDataGenerator.from_loader(loader, split=0).batch(BATCH_SIZE)
ds_val   = SegDataGenerator.from_loader(loader, split=1).batch(BATCH_SIZE)

mdl = build_seg_model(name=MODEL_NAME, is_logistic=True, glob=True)
mdl.summary()

# Compile model
mdl.compile(
	optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay=0.004),
	loss = mixed_loss, #weighted_ce_loss, #dice_loss,
	metrics = [
		logistic_adapter(tf.keras.metrics.BinaryAccuracy)(name='acc'),
		logistic_adapter(tf.keras.metrics.Precision)(name='prec'),
		logistic_adapter(tf.keras.metrics.Recall)(name='rcl'),
		logistic_adapter(tf.keras.metrics.AUC)(name='pr-auc', curve='PR'),
	],
)

train_args = {
	'x': ds_train,
	'validation_data': ds_val,
	'epochs': NUM_EPOCHS,
}

early_stopping = tf.keras.callbacks.EarlyStopping(
	monitor="val_loss", mode="min",
	patience=20,
	min_delta=0.001,
	restore_best_weights=True,
	verbose=True,
)

pretty_train(mdl, MODEL_NAME, early_stopping, **train_args)

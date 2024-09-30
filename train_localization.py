import tensorflow as tf
from utils.data_loader import DataLoader
from utils.localization import LocDataGenerator
from utils.models import build_loc_model, loc_model_loss
from utils.training import pretty_train

DATASET_FILE = 'frog_11-36_12-43_train_val.h5'

MODEL_NAME = 'LFE_PPN'
BACKBONE_MODEL = 'LFE_seg_20230421145111.h5'
NUM_EPOCHS = 100
BATCH_SIZE = 4

# Create data loader
loader = DataLoader(DATASET_FILE, min_people=3, points_per_sector=6)
print(len(loader),'scans in total')

ds_train = LocDataGenerator.from_loader(loader, split=0).batch(BATCH_SIZE)
ds_val   = LocDataGenerator.from_loader(loader, split=1).batch(BATCH_SIZE)

backbone = None
if BACKBONE_MODEL:
	backbone = tf.keras.models.load_model(BACKBONE_MODEL, compile=False).get_layer('backbone')

mdl = build_loc_model(name=MODEL_NAME, backbone=backbone, num_anchors_per_sector=loader.NUM_ANCHORS_PER_SECTOR, glob=True)
mdl.summary()

# Compile model
mdl.compile(
	optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001, weight_decay=0.0004),
	loss = loc_model_loss,
	metrics = [
		# XX: are there any metrics that make sense to calculate here?
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

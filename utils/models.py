import tensorflow as tf
import numpy as np

def logistic_adapter(clazz):
	class AdaptedMetric(clazz):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
		def update_state(self, y_true, y_pred, sample_weight=None):
			return super().update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight=sample_weight)
	return AdaptedMetric

def my_global_aggregator_block(x):
	x_global = tf.keras.layers.GlobalMaxPool1D(keepdims=True)(x)
	x_global = tf.broadcast_to(x_global, tf.shape(x))
	x = tf.keras.layers.Concatenate()([x, x_global])
	return x

def my_res_conv_block(x, filters, kernels, ac='relu', dp=0.2, bn=True, glob=False):
	x2 = x
	if x2.shape[-1] != filters[-1]:
		x2 = tf.keras.layers.Conv1D(filters=filters[-1], kernel_size=1, use_bias=False)(x)

	if glob:
		x = my_global_aggregator_block(x)

	for i, (f, k) in enumerate(zip(filters, kernels)):
		if i < len(filters)-1:
			x = tf.keras.layers.SeparableConv1D(filters=f, kernel_size=k, padding='same', activation=ac)(x)
			if bn: x = tf.keras.layers.BatchNormalization()(x)
			if dp > 0.0: x = tf.keras.layers.Dropout(dp)(x)
		else:
			x = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='same')(x)

	x = tf.keras.layers.Add()([x,x2])
	x = tf.keras.layers.Activation(ac)(x)
	if bn: x = tf.keras.layers.BatchNormalization()(x)
	if dp > 0.0: x = tf.keras.layers.Dropout(dp)(x)

	return x

def my_upscale_block(x, factor, ac='relu', dp=0.2, bn=True):
	#x = tf.keras.layers.Conv1DTranspose(filters=x.shape[-1], kernel_size=factor, strides=factor, padding='valid', activation=ac)(x)
	#if bn: x = tf.keras.layers.BatchNormalization()(x)
	#if dp > 0.0: x = tf.keras.layers.Dropout(dp)(x)
	x = tf.keras.layers.UpSampling1D(size=factor)(x)
	return x

def dice_loss(y_true, y_pred, smooth=1.0):
	num_TP       = tf.reduce_sum(y_true * y_pred, axis=[1,2])
	num_pos_gt   = tf.reduce_sum(y_true, axis=[1,2])
	num_pos_pred = tf.reduce_sum(y_pred, axis=[1,2])
	return 1.0 - (2.0 * num_TP + smooth) / (num_pos_gt + num_pos_pred + smooth)

def weighted_ce_loss(y_true, y_pred, beta=1.5):
	return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, tf.constant(beta, dtype=tf.float32))

def mixed_loss(y_true, y_pred):
	dice = dice_loss(y_true, tf.nn.sigmoid(y_pred))
	ce = weighted_ce_loss(y_true, y_pred)
	return 0.5*(dice + ce)

def loc_model_loss(y_true, y_pred):
	# Shape of y_true/y_pred: (BS, num_sectors, num_anchors_per_sector, 3) where:
	#   0=objectness (logits), 1=diff_dist, 2=diff_ang

	gt_objectness = y_true[:,:,:,0]
	gt_regression = y_true[:,:,:,1:3]

	pred_objectness = y_pred[:,:,:,0]
	pred_regression = y_pred[:,:,:,1:3]

	num_pos_anchors = tf.cast(tf.math.count_nonzero(gt_objectness, axis=[1,2]), tf.float32) + tf.keras.backend.epsilon()

	# Calculate "classification" loss
	cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(gt_objectness, pred_objectness)
	cls_loss = tf.reduce_mean(cls_loss, axis=[1,2])

	# Calculate regression loss
	## XX: "Smooth L1 loss", with sigma hyperparameter. See RPN source
	sigma_squared = 3.0*3.0
	diff = tf.math.abs(gt_regression - pred_regression)
	piecewise_branch = tf.stop_gradient(tf.cast(tf.less(diff, 1.0/sigma_squared), tf.float32))
	reg_loss_pos = 0.5*diff*diff*sigma_squared
	reg_loss_neg = diff - 0.5/sigma_squared
	reg_loss = piecewise_branch*reg_loss_neg + (1.0-piecewise_branch)*reg_loss_pos
	reg_loss = tf.reduce_sum(
		gt_objectness[:,:,:,None]*reg_loss,
		axis=[1,2]) / num_pos_anchors[:,None]
	reg_loss = tf.reduce_mean(reg_loss, axis=1)

	# Final loss
	return cls_loss + reg_loss

def loc_model_loss_with_mask(y_true, y_pred):
	# Shape of y_true: (BS, num_sectors, num_anchors_per_sector, 4) where:
	#   0=included, 1=objectness, 2=diff_dist, 3=diff_ang
	# Shape of y_pred: (BS, num_sectors, num_anchors_per_sector, 3)
	#   0=objectness logits, 1=diff_dist, 2=diff_ang

	gt_mask       = y_true[:,:,:,0]
	gt_objectness = y_true[:,:,:,1]
	gt_regression = y_true[:,:,:,2:4]

	pred_objectness = y_pred[:,:,:,0]
	pred_regression = y_pred[:,:,:,1:3]

	num_used_anchors = tf.cast(tf.math.count_nonzero(gt_mask, axis=[1,2]), tf.float32) + tf.keras.backend.epsilon()
	num_pos_anchors = tf.cast(tf.math.count_nonzero(gt_mask*gt_objectness, axis=[1,2]), tf.float32) + tf.keras.backend.epsilon()

	# Calculate "classification" loss
	cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(gt_objectness, pred_objectness)
	cls_loss = tf.reduce_sum(gt_mask*cls_loss, axis=[1,2]) / num_used_anchors

	# Calculate regression loss
	## XX: "Smooth L1 loss", with sigma hyperparameter. See RPN source
	## XX: Need to use gt_mask*gt_objectness as mask (we only care about regression loss in *positive* examples)
	sigma_squared = 3.0*3.0
	diff = tf.math.abs(gt_regression - pred_regression)
	piecewise_branch = tf.stop_gradient(tf.cast(tf.less(diff, 1.0/sigma_squared), tf.float32))
	reg_loss_pos = 0.5*diff*diff*sigma_squared
	reg_loss_neg = diff - 0.5/sigma_squared
	reg_loss = piecewise_branch*reg_loss_neg + (1.0-piecewise_branch)*reg_loss_pos
	reg_loss = tf.reduce_sum(
		tf.expand_dims(gt_mask*gt_objectness,axis=3)*reg_loss,
		axis=[1,2]) / num_pos_anchors[:,None]
	reg_loss = tf.reduce_mean(reg_loss, axis=1)

	# Final loss
	return cls_loss + reg_loss

def build_backbone(name=None, glob=False):
	# Input layer
	y = x = tf.keras.Input(shape=(None, 1))

	# Downscaling path
	y = y_1 = my_res_conv_block(y, filters=[32, 32, 32], kernels=[9, 7, 5])
	y = tf.keras.layers.MaxPool1D(pool_size=2)(y)
	y = y_2 = my_res_conv_block(y, filters=[32, 32, 32], kernels=[9, 7, 5], glob=glob)
	y = tf.keras.layers.MaxPool1D(pool_size=3)(y)
	y = my_res_conv_block(y, filters=[32, 32, 32], kernels=[9, 7, 5], glob=glob)

	return tf.keras.Model(inputs=x, outputs=[y, y_2, y_1], name=name)

def build_seg_model(name=None, backbone=None, is_logistic=False, glob=False):
	# Input layer
	y = x = tf.keras.Input(shape=(None, 1))

	# Downscaling path (backbone)
	if backbone is None:
		backbone = build_backbone('backbone', glob=glob)
	y_6, y_2, y_1 = backbone(y)

	# Upscaling path
	y = my_upscale_block(y_6, 3)
	y = tf.keras.layers.Concatenate()([y_2, y])
	y = my_res_conv_block(y, filters=[32, 32, 32], kernels=[9, 7, 5], glob=glob)
	y = my_upscale_block(y, 2)
	y = tf.keras.layers.Concatenate()([y_1, y])
	y = my_res_conv_block(y, filters=[32, 32, 32], kernels=[9, 7, 5], glob=glob)

	# Segmentation head
	y = tf.keras.layers.Conv1D(filters=1, kernel_size=1, padding='same', activation=None if is_logistic else 'sigmoid')(y)

	# Build model
	return tf.keras.Model(inputs=x, outputs=y, name=name)

def extract_seg_out(batch, is_logistic=True):
	if is_logistic: batch = tf.nn.sigmoid(batch)
	return batch[:,:,0].numpy()

def build_loc_model(name=None, backbone=None, num_anchors_per_sector=6, glob=False):
	# Input layer
	y = x = tf.keras.Input(shape=(None, 1))

	# Feature extractor (backbone)
	if backbone is None:
		backbone = build_backbone('backbone', glob=glob)
	y, _, _ = backbone(y)

	# Localization head
	if glob: y = my_global_aggregator_block(y)
	y = tf.keras.layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(y)
	y = tf.keras.layers.Conv1D(filters=3*num_anchors_per_sector, kernel_size=1, activation=None)(y)
	y = tf.keras.layers.Reshape((-1, num_anchors_per_sector, 3))(y)

	# Build model
	return tf.keras.Model(inputs=x, outputs=y, name=name)

def extract_loc_out(batch):
	out = np.empty(tf.shape(batch), dtype=np.float32)
	out[:,:,:,0]   = tf.nn.sigmoid(batch[:,:,:,0]).numpy()
	out[:,:,:,1:3] = batch[:,:,:,1:3].numpy()
	return out

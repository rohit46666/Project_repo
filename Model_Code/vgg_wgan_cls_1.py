import tensorflow as tf
from scipy.misc import imshow
from numpy import *
import tensorflow.contrib.slim as slim
from vgg import *

TRAIN_FILE = "/home/sujit/Desktop/coco/train.tfrecords"
VALIDATION_FILE = "/home/sujit/Desktop/coco/val.tfrecords"

batch = 32
epochs = 20

#Input 

def decode(serialized_example):
	features = tf.parse_single_example(
		serialized_example,
		features = {
			"image_raw": tf.FixedLenFeature([], tf.string),
			"captions" : tf.FixedLenFeature([], tf.string),
			"label" : tf.FixedLenFeature([], tf.string)
		})

	image = tf.decode_raw(features["image_raw"], tf.float32)
	image = tf.reshape(image, shape = [224,224,3])

	captions = tf.decode_raw(features["captions"], tf.float64)
	captions = tf.cast(captions, tf.float32)
	captions = tf.reshape(captions, shape = [3000])

	label = tf.decode_raw(features["label"], tf.float64)
	label = tf.cast(label, tf.float32)
	label = tf.reshape(label, shape=[90])

	return image, captions, label


def normalize(image, captions, label):
	image = image * (1./255)
	return image, captions, label


def inputs(train, batch_size, num_epochs):

	filename = TRAIN_FILE if train else VALIDATION_FILE

	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(decode)
	dataset = dataset.map(normalize)


	dataset = dataset.shuffle(1000 + 3*batch_size)

	dataset = dataset.repeat(num_epochs)
	dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))


	iterator = dataset.make_one_shot_iterator()
	
	return iterator.get_next()


###########################################################################################################################################





## Network Layers

def commonTextEnc(input):
	with tf.variable_scope("commonText", reuse = tf.AUTO_REUSE) as scope:
		ct0 = tf.nn.leaky_relu(tf.layers.dense(input, 2000))
		ct1 = tf.nn.leaky_relu(tf.layers.dense(ct0, 1000))
		return ct1

def textCont(input):
	with tf.variable_scope("textCont", reuse = tf.AUTO_REUSE) as scope:
		tc1 = tf.nn.leaky_relu(tf.layers.dense(input, 512))
		tc2 = tf.nn.leaky_relu(tf.layers.dense(tc1, 512))
		return tc2

def commonImgEnc(input):
	with tf.variable_scope("commonImage", reuse = tf.AUTO_REUSE) as scope:
		ci0 = tf.nn.leaky_relu(tf.layers.conv2d(input, name='ci0', filters = 128, kernel_size=5, strides=2, padding="SAME"))
		ci1 = tf.nn.leaky_relu(tf.layers.conv2d(ci0, name='ci1', filters = 128, kernel_size=5, strides=1, padding="SAME"))
		ci2 = tf.nn.leaky_relu(tf.layers.conv2d(ci1, name='ci2', filters = 128, kernel_size=5, strides=2, padding="SAME"))
		ci3 = tf.nn.leaky_relu(tf.layers.conv2d(ci2, name='ci3', filters = 256, kernel_size = 3, strides = 1, padding="SAME"))
		ci4 = tf.nn.leaky_relu(tf.layers.conv2d(ci3, name='ci4', filters = 256, kernel_size = 3, strides = 2, padding="SAME"))
		return ci4


def imgCont(input):
	with tf.variable_scope("imgCont", reuse = tf.AUTO_REUSE) as scope:
		ic4 = tf.nn.leaky_relu(tf.layers.conv2d(input, name="ic4", filters = 512, kernel_size = 3, strides=1, padding = "SAME"))
		ic4_1 = tf.nn.leaky_relu(tf.layers.conv2d(ic4, name="ic4_1", filters=512, kernel_size = 3, strides=2, padding = "SAME"))
		ic4_2 = tf.nn.leaky_relu(tf.layers.conv2d(ic4_1, name="ic4_2", filters=1024, kernel_size = 3, strides = 2, padding = "SAME"))
		ic5 = tf.layers.flatten(ic4_2, name="ic5")
		# print(ic5)
		ic5_1 = tf.nn.leaky_relu(tf.layers.dense(ic5, 2048))
		ic6 = tf.nn.leaky_relu(tf.layers.dense(ic5_1, 1024))
		ic7 = tf.nn.leaky_relu(tf.layers.dense(ic6, 512))
		return ic7


def sharedCont(input):
	with tf.variable_scope("sharedCont", reuse = tf.AUTO_REUSE) as scope:
		sc0 = tf.nn.leaky_relu(tf.layers.dense(input, 512))
		sc1 = tf.nn.leaky_relu(tf.layers.dense(sc0, 350))
		sc2 = tf.nn.leaky_relu(tf.layers.dense(sc1, 256))
		return sc2 / tf.norm(sc2, axis=1, keep_dims=True)


def classifier(input):
	with tf.variable_scope("classifier", reuse = tf.AUTO_REUSE) as scope:
		cl0 = tf.nn.leaky_relu(tf.layers.dense(input, 256))
		cl1 = tf.nn.leaky_relu(tf.layers.dense(input, 128))
		cl2 = tf.nn.leaky_relu(tf.layers.dense(cl1, 100))
		cl3 = tf.layers.dense(cl2, 90)
		return cl3


def association_same(input, label):
	# pairwise cosine distance
	normalized_input = input / tf.norm(input, axis=1, keep_dims=True)
	cosine_distance = tf.matmul(normalized_input,normalized_input, adjoint_b=True)
	zers = tf.zeros(batch)
	n_inf = tf.log(zers)
	cosine_distance = tf.reshape(cosine_distance, [-1])


	# pairwise jacard index for labels
	expanded_label_dims = tf.expand_dims(label, axis=1)
	tiled = tf.tile(expanded_label_dims, (1,batch,1))
	transpose = tf.transpose(tiled, perm=[1,0,2])
	stacked = tf.stack([tiled, transpose])
	minimum = tf.reduce_min(stacked, axis=0)
	maximum = tf.reduce_max(stacked, axis=0)
	min_sum = tf.reduce_sum(minimum, axis=2)
	max_sum = tf.reduce_sum(maximum, axis=2)
	max_sum = tf.where(tf.less(max_sum, 1e-7), 1e-7*tf.ones_like(max_sum),max_sum)
	jacard = tf.divide(min_sum, max_sum)
	jacard = tf.reshape(jacard, [-1])
	mean_n_std = tf.multiply(tf.ones_like(jacard), 0.5)

	jacard = tf.divide(tf.subtract(jacard, mean_n_std), mean_n_std)

	distance = tf.multiply(jacard, cosine_distance)

	return  tf.reduce_mean(distance)

	# return tf.nn.softmax_cross_entropy_with_logits(logits = cosine_distance, labels = tf.nn.softmax(jacard, axis=1))


def gradient_penalty(real, fake, f):
	def interpolate(a, b):
		alpha = tf.random_uniform(shape=[batch,1], minval=0., maxval=1.)
		inter = a + alpha * (b - a)
		return inter
	x = interpolate(real, fake)
	pred = f(x)
	gradients = tf.gradients(pred, x)[0]
	slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
	gp = tf.reduce_mean((slopes - 1.)**2)
	return gp


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise


def image_augmentation(input):
	radians = tf.multiply(tf.random_normal(shape=tf.shape(batch), mean=180, stddev=180, dtype=tf.float32), math.pi/180)
	return tf.contrib.image.rotate(input, radians)


def pairwise_concat(input1, input2, label1, label2):
	#generate n similar pairs
	# input1 = input1 / tf.norm(input1, axis=1, keep_dims=True)
	# input2 = input2 / tf.norm(input2, axis=1, keep_dims=True)
	pairs_sim = tf.concat([input1, input2], 1)

	#generate n diff pairs
	pairs_diff = tf.concat([input1, tf.reverse(input2, [0])], 1)


	#generate n^2 diff pairs
	expanded1 = tf.expand_dims(input1, axis=1)
	expanded2 = tf.expand_dims(input2, axis=1)
	til1 = tf.tile(expanded1, (1, batch, 1))
	til2 = tf.tile(expanded2, (1, batch, 1))
	transpose = tf.transpose(til2, perm=[1, 0, 2])
	concat = tf.concat([til1, transpose], 2)
	mask = tf.expand_dims(1. - tf.eye(batch), -1)
	mask = tf.tile(mask, (1, 1, 512))
	non_diag_indices = tf.where(tf.equal(mask, 1.))
	non_diag_elements = tf.gather_nd(concat, non_diag_indices)
	all_pairs_diff = tf.reshape(non_diag_elements, [-1, 512])


	#generate n^2 jacard index
	expanded_label1_dims = tf.expand_dims(label1, axis = 1)
	expanded_label2_dims = tf.expand_dims(label2, axis = 1)
	tiled1 = tf.tile(expanded_label1_dims, (1, batch, 1))
	tiled2 = tf.tile(expanded_label2_dims, (1, batch, 1))
	transposed = tf.transpose(tiled2, perm=[1, 0, 2])
	stacked = tf.stack([tiled1, transposed])
	minimum = tf.reduce_min(stacked, axis = 0)
	maximum = tf.reduce_max(stacked, axis = 0)
	min_sum = tf.reduce_sum(minimum, axis = 2)
	max_sum = tf.reduce_sum(maximum, axis = 2)
	max_sum = tf.where(tf.less(max_sum, 1e-7), 1e-7*tf.ones_like(max_sum),max_sum)
	jacard = tf.divide(min_sum, max_sum)
	mask = 1. - tf.eye(batch)
	non_diag_indices = tf.where(tf.equal(mask, 1.))
	non_diag_elements = tf.gather_nd(jacard, non_diag_indices)
	all_pairs_jacard = tf.reshape(non_diag_elements, [-1, 1])

	return pairs_sim, pairs_diff, all_pairs_diff, all_pairs_jacard


def pairwise_concat_diff(input1, input2, label1, label2):
	expanded1 = tf.expand_dims(input1, axis=1)
	expanded2 = tf.expand_dims(input2, axis=1)
	til1 = tf.tile(expanded1, (1, batch, 1))
	til2 = tf.tile(expanded2, (1, batch, 1))
	transpose = tf.transpose(til2, perm=[1, 0, 2])
	concat = tf.concat([til1, transpose], 2)
	pairs = tf.reshape(concat, [-1, 512])

	expanded_label1_dims = tf.expand_dims(label1, axis=1)
	expanded_label2_dims = tf.expand_dims(label2, axis=1)
	tiled1 = tf.tile(expanded_label1_dims, (1, batch, 1))
	tiled2 = tf.tile(expanded_label2_dims, (1, batch, 1))
	transposed = tf.transpose(tiled2, perm=[1, 0, 2])
	stacked = tf.stack([tiled1, transposed])
	minimum = tf.reduce_min(stacked, axis=0)
	maximum = tf.reduce_max(stacked, axis=0)
	min_sum = tf.reduce_sum(minimum, axis=2)
	max_sum = tf.reduce_sum(maximum, axis=2)
	max_sum = tf.where(tf.less(max_sum, 1e-7), 1e-7*tf.ones_like(max_sum),max_sum)
	jacard = tf.reshape(tf.divide(min_sum, max_sum), [-1, 1])

	return pairs, jacard

def similarityDisc(input):
	with tf.variable_scope("similarity_disc", reuse = tf.AUTO_REUSE) as scope:
		sd0 = tf.nn.leaky_relu(tf.layers.dense(input, 256))
		sd1 = tf.nn.leaky_relu(tf.layers.dense(sd0, 128))
		sd2 = tf.nn.leaky_relu(tf.layers.dense(sd1, 64))
		sd3 = tf.nn.leaky_relu(tf.layers.dense(sd2, 8))
		sd4 = tf.layers.dense(sd3, 1)
		return sd4
















## Architecture #####################################################################################################

image_batch, caption_batch, label_batch = inputs(train=False, batch_size = batch, num_epochs = epochs)
aug_image_batch = image_augmentation(image_batch)
aug_caption_batch = gaussian_noise_layer(caption_batch, 0.02)
output = tf.sign(label_batch)














## Text architecture and Losses #######################################################################################

com_txt = commonTextEnc(caption_batch)
txt_cont = textCont(com_txt)
shared_txt_cont = sharedCont(txt_cont)


a_com_txt = commonTextEnc(aug_caption_batch)
a_txt_cont = textCont(a_com_txt)
a_shared_txt_cont = sharedCont(a_txt_cont)

txt_pairs_sim, txt_pairs_diff, all_txt_pairs_diff, all_txt_pairs_diff_jacard = pairwise_concat(shared_txt_cont, a_shared_txt_cont, output, output)


txt_cls = classifier(shared_txt_cont)

txt_sim_disc = similarityDisc(txt_pairs_sim)
txt_diff_disc = similarityDisc(txt_pairs_diff)
all_txt_disc = similarityDisc(all_txt_pairs_diff)

wd_txt = tf.reduce_mean(txt_sim_disc) - tf.reduce_mean(txt_diff_disc)

gp_txt = gradient_penalty(txt_pairs_sim, txt_pairs_diff, similarityDisc)

txt_sim_disc_loss = -wd_txt + (gp_txt * 10.0)

txt_asc_loss = -(tf.reduce_mean(tf.multiply(all_txt_disc, all_txt_pairs_diff_jacard)))
txt_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=txt_cls, labels=output))
# total_txt_loss = txt_asc_loss + txt_cls_loss






## Image Architecture and Losses ########################################################################################




image_vgg_c13 = vgg_16(image_batch)
com_img = image_vgg_c13[1]["vgg_16/conv5/conv5_3"]
img_cont = imgCont(com_img)
shared_img_cont = sharedCont(img_cont)

a_image_vgg_c13 = vgg_16(aug_image_batch)
a_com_img = a_image_vgg_c13[1]["vgg_16/conv5/conv5_3"]
a_img_cont = imgCont(a_com_img)
a_shared_img_cont = sharedCont(a_img_cont)




img_pairs_sim, img_pairs_diff, all_img_pairs_diff, all_img_pairs_diff_jacard = pairwise_concat(shared_img_cont, a_shared_img_cont, output, output)

img_sim_disc = similarityDisc(img_pairs_sim)
img_diff_disc = similarityDisc(img_pairs_diff)
all_img_disc = similarityDisc(all_img_pairs_diff)

wd_img = tf.reduce_mean(img_sim_disc) - tf.reduce_mean(img_diff_disc)

gp_img = gradient_penalty(img_pairs_sim, img_pairs_diff, similarityDisc)

img_sim_disc_loss = -wd_img + (gp_img*10.0)

img_asc_loss = -(tf.reduce_mean(tf.multiply(all_img_disc, all_img_pairs_diff_jacard)))

img_cls = classifier(shared_img_cont)

# img_asc_loss = -association_same(shared_img_cont, label_batch)
img_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=img_cls, labels=output))
total_img_loss = img_asc_loss + img_cls_loss




## Total Loss ##########################################################################################################

# total_loss = total_img_loss + total_txt_loss
total_asc_loss = txt_asc_loss + img_asc_loss
total_sim_disc_loss = txt_sim_disc_loss + img_sim_disc_loss
total_cls_loss = txt_cls_loss + img_cls_loss




## Optimizers ############################################################################################################


asc_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(total_asc_loss)
sim_disc_op = tf.train.AdamOptimizer().minimize(total_sim_disc_loss)
cls_op = tf.train.AdamOptimizer().minimize(total_cls_loss)


































init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



with tf.Session() as sess:
	sess.run(init_op)
	
	variables_to_restore = slim.get_variables_to_restore(include=["vgg_16"])
	restore_vars = [v for v in variables_to_restore if not("Adam" in v.name)]
	init_assign_op, init_feed_dict = slim.assign_from_checkpoint("vgg_16.ckpt", restore_vars)
	sess.run(init_assign_op, init_feed_dict)

	# print(sess.run(predictions[1]["vgg_16/conv5/conv5_3"]).shape)

	try:
		step = 0
		while True:
			_, asc_l, i_asc, t_asc = sess.run([asc_op, total_asc_loss, img_asc_loss, txt_asc_loss])
			_, s_d_l, i_d_l, t_d_l = sess.run([sim_disc_op, total_sim_disc_loss, img_sim_disc_loss, txt_sim_disc_loss])
			_, c_l, i_c_l, t_c_l = sess.run([cls_op, total_cls_loss, img_cls_loss, txt_cls_loss])
			step = step + 1
			print('Step %d: asc_l = %.9f, i_asc = %.9f, t_asc = %.9f, s_d_l = %.9f, i_d_l = %.9f, t_d_l = %.9f, c_l = %.9f, i_c_l = %.9f, t_c_l = %.9f' % (step, 
				asc_l, i_asc, t_asc, s_d_l, i_d_l, t_d_l, c_l, i_c_l, t_c_l))
			
	except tf.errors.OutOfRangeError:
		print("Done training for %d epochs, %d steps." % (epochs, step))
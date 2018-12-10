import tensorflow as tf
from scipy.misc import imshow
from numpy import *
import numpy as np
import tensorflow.contrib.slim as slim
from vgg import *
from PIL import Image
TRAIN_FILE = "/home/rohit/Project_scrip/Source/train_demo.tfrecords"
VALIDATION_FILE = "/home/rohit/Project_scrip/Source/val_demo.tfrecords"

batch = 10
epochs = 2
IMAGE_SIZE = 224
#Input 

def decode(serialized_example):
	features = tf.parse_single_example(
		serialized_example,
		features = {
			"image_raw": tf.FixedLenFeature([], tf.string),
			"label" : tf.FixedLenFeature([], tf.float32)
		})

	image = tf.decode_raw(features["image_raw"], tf.uint8)
	image = tf.reshape(image, shape = [IMAGE_SIZE,IMAGE_SIZE,3])
	image = tf.cast(image, tf.float32)


	# label = tf.decode_raw(features["label"], tf.float64)
	
	label = tf.reshape(features["label"], shape=[1])
	label = tf.cast(label, tf.float32)

	return image,label


def normalize(image, label):
	image = image * (1./255)

	return image, label


def inputs(train, batch_size, num_epochs,fileName1):

	filename = fileName1
	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(decode)
	# dataset = dataset.map(normalize)


	# dataset = dataset.shuffle(10+ 3*batch_size)

	dataset = dataset.repeat(num_epochs)
	dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))


	iterator = dataset.make_one_shot_iterator()
	return iterator.get_next()
	


###########################################################################################################################################

## Network Layers


def classifier(input):
	with tf.variable_scope("classifier", reuse = tf.AUTO_REUSE) as scope:
		# print type(input)
		cl0 = tf.nn.leaky_relu(tf.layers.dense(tf.contrib.layers.flatten(input), 256))
		cl1 = tf.nn.leaky_relu(tf.layers.dense(cl0, 128))
		cl2 = tf.nn.leaky_relu(tf.layers.dense(cl1, 100))
		cl3 = tf.layers.dense(cl2, 1)
		return cl3




## Architecture #####################################################################################################
#  batch of images and output
image_batch,output = inputs(train=True, batch_size = batch, num_epochs = epochs,fileName1 = TRAIN_FILE)

## Text architecture and Losses #######################################################################################



## Image Architecture and Losses ########################################################################################

image_vgg_c13 = vgg_16(image_batch)
 # print type(image_vgg_c13)

img_cls = classifier(image_vgg_c13[1]["vgg_16/conv5/conv5_3"])
# # print ("!!!!!!!!!!!!!!!!!!!!!1predicted value",img_cls)
pred = tf.sigmoid(img_cls)
loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=img_cls, labels=output)
img_cls_loss = tf.reduce_mean(loss_vec)

## validation Code #############################################
def Validation():
	val_image,val_output = inputs(train=True, batch_size = batch, num_epochs = epochs,fileName1 = VALIDATION_FILE)
	image_vgg_c13 = vgg_16(val_image)
	val_img_cls = classifier(image_vgg_c13[1]["vgg_16/conv5/conv5_3"])
	pred = tf.sigmoid(val_img_cls)
	t= tf.where(tf.greater_equal(pred,tf.constant(0.5)),tf.ones_like(pred),tf.zeros_like(pred))
	accuracy = tf.equal(tf.cast(t,tf.int32),tf.cast(val_output,tf.int32))
	# accuracy = tf.divide(tf.reduce_sum(v),batch)
	return accuracy
 	


## Total Loss ##########################################################################################################





## Optimizers ############################################################################################################



cls_op = tf.train.AdamOptimizer(.00005).minimize(img_cls_loss)

## session run

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

acc = Validation()


with tf.Session() as sess:
	sess.run(init_op)
	Tp = 0.0
	Counter = 0.0
	variables_to_restore = slim.get_variables_to_restore(include=["vgg_16"])
	# print variables_to_restore
	restore_vars = [v for v in variables_to_restore if not("Adam" in v.name)]
	init_assign_op, init_feed_dict = slim.assign_from_checkpoint("vgg_16.ckpt", restore_vars)
	sess.run(init_assign_op, init_feed_dict)
	step = 0
	print "Weight updated and reading"
	# print(sess.run(predictions[1]["vgg_16/conv5/conv5_3"]).shape)
	try :
		while True :
			a,b,c,d,e =sess.run([image_batch,img_cls,pred,img_cls_loss,cls_op])
			# print ("True label for batch",a)
			step = step + 1
			# print ("predicted lable probabilty by sigmod",c)
			print('Step %d loss %f' %(step,d))
			# print(" Image Shape" , e.shape)
			# Image.fromarray(np.asarray(e)).show()
	except tf.errors.OutOfRangeError:
		print (" Training Done")
		print ("Start Testing")
		try :
			while True :
				boollabel = sess.run([acc])
				Tp = Tp + np.sum(boollabel)
				Counter = Counter + batch
				print('True Postive    Total counter  ',Tp,Counter)
		except tf.errors.OutOfRangeError:
			# print ("Accuracy %f", Tp/Counter)
			print("Done Validation" )
			
			


		

			

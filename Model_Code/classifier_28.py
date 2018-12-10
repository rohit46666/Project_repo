import tensorflow as tf
from numpy import *
import numpy as np
import tensorflow.contrib.slim as slim
from vgg import *
from PIL import Image

TRAIN_FILE = "/home/rohit/Project_scrip/Source/train_demo.tfrecords"
VALIDATION_FILE = "/home/rohit/Project_scrip/Source/val_demo.tfrecords"
TEST_FILE ="/home/rohit/Project_scrip/Source/test_demo.tfrecords"

batch = 16
epochs = 8
IMAGE_SIZE = 224


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

	label = tf.reshape(features["label"], shape=[1])
	label = tf.cast(label, tf.float32)

	return image,label


def normalize(image, label):
	image = image * (1./255)
	return image, label


def inputs(batch_size, num_epochs,fileName):

	dataset = tf.data.TFRecordDataset(fileName)
	dataset = dataset.map(decode)
	dataset = dataset.map(normalize)

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
		cl2 = tf.nn.leaky_relu(tf.layers.dense(cl1, 64))
		cl3 = tf.nn.leaky_relu(tf.layers.dense(cl2, 32))
		cl5 = tf.layers.dense(cl3, 1)
		return cl5




## Architecture #####################################################################################################

#  batch of images and output

image_batch,output = inputs(batch_size = batch, num_epochs = epochs,fileName = TRAIN_FILE)


## Image Architecture and Losses ########################################################################################


image_vgg_c13 = vgg_16(image_batch)
img_cls = classifier(image_vgg_c13[1]["vgg_16/conv5/conv5_3"])
loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=img_cls, labels=output)
img_cls_loss = tf.reduce_mean(loss_vec)

## Optimizers ############################################################################################################


cls_op = tf.train.AdamOptimizer(.00005).minimize(img_cls_loss)


## validation Code #############################################

def Validation():
	val_image,val_output = inputs(batch_size = 1, num_epochs = 1,fileName = VALIDATION_FILE)
	image_vgg_c13 = vgg_16(val_image)
	val_img_cls = classifier(image_vgg_c13[1]["vgg_16/conv5/conv5_3"])
	pred = tf.sigmoid(val_img_cls)
	t= tf.where(tf.greater_equal(pred,tf.constant(0.5)),tf.ones_like(pred),tf.zeros_like(pred))
	accuracy = tf.equal(tf.cast(t,tf.int32),tf.cast(val_output,tf.int32))
	# accuracy = tf.divide(tf.reduce_sum(v),batch)
	return accuracy
 	

val_acc = Validation()


## Testing Code #############################################

def Testing():
	test_image,test_output = inputs(batch_size = 1, num_epochs = 1,fileName = TEST_FILE)
	image_vgg_c13 = vgg_16(test_image)
	test_img_cls = classifier(image_vgg_c13[1]["vgg_16/conv5/conv5_3"])
	pred = tf.sigmoid(test_img_cls)
	t= tf.where(tf.greater_equal(pred,tf.constant(0.5)),tf.ones_like(pred),tf.zeros_like(pred))
	accuracy = tf.equal(tf.cast(t,tf.int32),tf.cast(test_output,tf.int32))
	
	return accuracy
 	

Test_acc = Testing()

## session run

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



with tf.Session() as sess:
	sess.run(init_op)
	Tp = 0.0
	Counter = 0.0
	loss = 0.0
	ep = 1
	variables_to_restore = slim.get_variables_to_restore(include=["vgg_16"])

	restore_vars = [v for v in variables_to_restore if not("Adam" in v.name)]
	init_assign_op, init_feed_dict = slim.assign_from_checkpoint("vgg_16.ckpt", restore_vars)
	sess.run(init_assign_op, init_feed_dict)
	step = 0
	print "Weight updated and reading"

	try :
		while True :
			d,e = sess.run([img_cls_loss,cls_op])
			step = step + 1
			loss = loss + d
			if step == 148 :
				print(' Epoch  %d  Per Epoch loss %f' %(ep,loss))
				step = 0
				loss = 0.0
				ep = ep + 1

				
	except tf.errors.OutOfRangeError:
		print(' Epoch  %d  Per Epoch loss %f' %(ep,loss))
		print (" Training Done")
		Tp = 0.0
		Counter = 0.0
		
		try :
			while True :
				boollabel = sess.run([val_acc])
				Tp = Tp + np.sum(boollabel)
				Counter = Counter + 1
				if Counter%50 == 0:
					print('True Postive    Total counter  ',Tp,Counter)
		except tf.errors.OutOfRangeError:
			print 'Accuracy :  {}/{}'.format( Tp,Counter)
			print "Done Validation" 
			print " Start Testing"
			test_count = 0.0
			Test_pred = 0.0
			try :
				while True :
						boollabel = sess.run([Test_acc])
						Test_pred = Test_pred + np.sum(boollabel)
						test_count = test_count + 1
						print('True Postive    Total counter  ',Test_pred,test_count)

			except tf.errors.OutOfRangeError :
				print ' Test_Accuracy :  {}/{}'.format( Test_pred,test_count)
			
			
			


		

			

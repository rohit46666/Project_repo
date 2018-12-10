from random import shuffle
import tensorflow as tf
import glob
import sys
import numpy as np
from PIL import Image
import random
import scipy.ndimage


IMAGE_SIZE = 224

shuffle_data = True  # shuffle the addresses before saving

train_path1 = '/home/rohit/New_DataSet/NewTempDataSet/Postive_data/*.jpeg'
addrs1 = glob.glob(train_path1)
labels1 = [1 for addr in addrs1]  # 1 =  Gun

train_path2 = '/home/rohit/New_DataSet/NewTempDataSet/Negative_data/*.jpeg'
addrs2 = glob.glob(train_path2)
labels2 = [0 for addr in addrs2]  # 0 = no gun


addrs = addrs1[:3500] + addrs2[:3500] 
labels = labels1[:3500] + labels2[:3500] 

test_path_1= "/home/rohit/New_DataSet/Test_dataSet/No_Gun/*.jpg"
test_addrs_1 = glob.glob(test_path_1)
test_label_1 = [0 for addr in test_addrs_1]

test_path_2= "/home/rohit/New_DataSet/Test_dataSet/Gun/*.jpg"
test_addrs_2 = glob.glob(test_path_2)
test_label_2 = [1 for addr in test_addrs_2]

print "started"
print ("Postive Sample Count : %d and labels count :%d ",len(addrs1),len(labels1))
print ("Negative Sample Count : %d and labels count :%d ",len(addrs2),len(labels2))

print ("Postive Sample Test Count : %d and labels count :%d ",len(test_addrs_2),len(test_label_2))
print ("Negative Sample Test Count : %d and labels count :%d ",len(test_addrs_1),len(test_label_1))

#Too shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)


# # Divide the hata into 80% train, 20% validation
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]
val_addrs = addrs[int(0.8*len(addrs)):]
val_labels = labels[int(0.8*len(addrs)):]


def load_image(addr):
	im = Image.open(addr)
	imn = np.array(im)
	imp = imn.tostring()

	return imp

def _int64_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Construct Train Tf Records
train_filename = 'train_demo.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename) # open the TFRecords file

for i in range(len(train_addrs)):
    if not i % 500:
        print 'Train data: {}/{}'.format(i, len(train_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]

    # Create a feature
    feature = {'image_raw': _bytes_feature(img),
               'label': _int64_feature(label)}
    
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush() 

# Construct Validation Records

Validation_fileName = 'val_demo.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(Validation_fileName) # open the TFRecords file
for i in range(len(val_addrs)):
    if not i % 500:
        print 'Train data: {}/{}'.format(i, len(val_addrs))
        sys.stdout.flush()

    # Load the image
    img = load_image(val_addrs[i])
    label = val_labels[i]

    # Create a feature
    feature = {'image_raw': _bytes_feature(img),
               'label': _int64_feature(label)}
    
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush() 

# test_Addrs = test_addrs_1 +test_addrs_2
# test_Label = test_label_1 +test_label_2

# #Too shuffle data
# if shuffle_data:
#     c = list(zip(test_Addrs, test_Label))
#     shuffle(c)
#     test_Addrs, test_Label = zip(*c)

# # Construct Test Records

# test_fileName = 'test_demo.tfrecords'  # address to save the TFRecords file
# writer = tf.python_io.TFRecordWriter(test_fileName) # open the TFRecords file
# for i in range(len(test_Addrs)):
#     if not i % 10:
#         print 'Train data: {}/{}'.format(i, len(test_Addrs))
#         sys.stdout.flush()

#     # Load the image
#     img  = load_image(test_Addrs[i])
#     label = test_Label[i]

# 	    # Create a feature
# 	feature = {'image_raw': _bytes_feature(img),
# 	               'label': _int64_feature(label)}
	    
# 	    # Create an example protocol buffer
# 	example = tf.train.Example(features=tf.train.Features(feature=feature))
	    
# 	    # Serialize to string and write on the file
# 	writer.write(example.SerializeToString())
    
# writer.close()
# sys.stdout.flush()             
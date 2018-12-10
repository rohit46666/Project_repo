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

train_path1 = '/home/rohit/New_DataSet/Pistol/*.jpg'
# read addresses and labels from the 'train' folder
addrs1 = glob.glob(train_path1)


# train_path2 = '/home/rohit/MObile/SHAREit/pictures/WhatsApp Images/*.jpg'
# # read addresses and labels from the 'train' folder
# addrs2 = glob.glob(train_path2)
# # print addrs


train_path3 = '/home/rohit/Downloads/google-images-download-master/google_images_download/downloads/action movie screenshot snapshot/*.jpg'
# read addresses and labels from the 'train' folder
addrs3 = glob.glob(train_path3)


addrs =  addrs3

print "started"
print ("Postive Sample Count : %d  ",len(addrs1))
print ("Negative Sample Count : %d  ",len(addrs3))
# to shuffle data
# if shuffle_data:
#     c = list(zip(addrs, labels))
#     shuffle(c)
#     addrs, labels = zip(*c)

# for i in range(2):
for i in range(len(addrs)):
	im = Image.open(addrs[i])
	height, width, channels = scipy.ndimage.imread(addrs[i]).shape
	flag = height >= 224 and width>=224 and channels ==3 
	if flag : 
		img = im.resize((IMAGE_SIZE,IMAGE_SIZE),Image.ANTIALIAS)
		rotated_1 = img.rotate(90)
		Fliped_Image_LR = img.transpose(Image.FLIP_LEFT_RIGHT)
		Fliped_Image_TB = img.transpose(Image.FLIP_TOP_BOTTOM)
		# img.show()
		img.save("/home/rohit/New_DataSet/NewTempDataSet/Negative_data/"+str(i)+".jpeg","JPEG")
		rotated_1.save("/home/rohit/New_DataSet/NewTempDataSet/Negative_data/"+"90_rotation"+str(i)+".jpeg","JPEG")
		Fliped_Image_LR.save("/home/rohit/New_DataSet/NewTempDataSet/Negative_data/"+"Fliped_Image_LR"+str(i)+".jpeg","JPEG")
		Fliped_Image_TB.save("/home/rohit/New_DataSet/NewTempDataSet/Negative_data/"+"Fliped_Image_TB"+str(i)+".jpeg","JPEG")
		if not i % 100:
			print 'Processed Image: {}/{}'.format(i, len(addrs))
			sys.stdout.flush()

	else :
		print " Found small size image"		

# for i in range(len(addrs1)):
# 	im = Image.open(addrs1[i])
# 	height, width, channels = scipy.ndimage.imread(addrs1[i]).shape
# 	flag = height >= 224 and width>=224 and channels ==3 
# 	if flag : 
# 		img = im.resize((IMAGE_SIZE,IMAGE_SIZE),Image.ANTIALIAS)
# 		# img.show()
# 		img.save("/home/rohit/New_DataSet/NewTempDataSet/Postive_data/"+str(i)+".jpeg","JPEG")
# 		if not i % 500:
# 			print 'Processed Image: {}/{}'.format(i, len(addrs1))
# 			sys.stdout.flush()

# 	else :
# 		print " Found small size image"	
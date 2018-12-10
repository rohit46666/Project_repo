import csv
import ast
import os
from PIL import Image 

# counter = 0
# dictionary = dict()
# input_file = 'file.csv'
# with open(input_file,'rb') as f:
# 	rdr = csv.DictReader(f)
# 	for row in rdr:
# 		# print type(row['first name'])
# 		# print row['first name']

# 		for dict_element in ast.literal_eval(row['images']):
# 			dictionary[(dict_element['path']).split('/')[1]] = dict_element['url'].split('/')[-1]
# 			counter +=1
# print type(dictionary.values()[0])
# print counter
# print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

path = '/home/rohit/DataSet/full'
files = os.listdir(path)
# idx = files.index("ccf58a2d3abb10e57333289fb0dafda801d77bd6.jpg")

# # print idx
# del files[idx]
print files
print len(files)

counter = 0
# Logic for Renming Files
# for f in files:
# 	counter +=1
# 	src = path+"/"+f
# 	print src
# 	# print type(dictionary[f])
# 	print (dictionary[f])
# 	print counter
# 	os.rename(src,path+"/"+dictionary[f])


# Logic for size Estimation 
size_matrix = dict()
counter = 0
for f in files:
	
	src = path+"/"+f
	# print src
	# print type(dictionary[f])
	im = Image.open(src)
	a,b = im.size
	c = str(a)+"*"+str(b)
	if c in size_matrix.keys():
		size_matrix[c] +=1
	else:
		 size_matrix[c] =1

		
	counter +=1	
	print counter 
	
print size_matrix		 


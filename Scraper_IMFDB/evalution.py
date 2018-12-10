# coding=utf-8
import csv
import codecs
import re
import ast
import os
from parse_Dict import csv_reader,parse_Url,search
from constructDict import creatCustomizeDict,creatCustomizeDict_baseForm
from shutil import copyfile
from collections import defaultdict

input_Dict = creatCustomizeDict_baseForm()	
# print input_Dict

output_counter = defaultdict(list)
output_map_image = dict()

input_file = 'Genrated_file.csv'

with open(input_file,'rb') as f:
	rdr = csv.DictReader(f)
	for row in rdr:
		# print type(row['first name'])
		# print row['first name']

		for dict_element in ast.literal_eval(row['images']):
			output_map_image[dict_element['path']] = dict_element['url']

	print output_map_image

	print type(output_map_image)

	for url,row in output_map_image.items():
			# print row[0]
		parsed_name = parse_Url(row)
			# print parsed_name
			# print ("parsed name type ",parsed_name)
		flag,key =search(input_Dict,parsed_name)
		if flag :
			output_counter[key].append(url)

print output_counter['Assault_Rifle']
print "DONE MApping"
src_dir = "/home/rohit/DataSet/"	
dst_dir = "/home/rohit/New_DataSet/"
print len (output_counter.keys())

print "Start copying ............."
for key,v in output_counter.items():
	print key
	print type(v)
	print len(v) 
	if not os.path.exists(dst_dir + key):
		os.makedirs(dst_dir + key)
	for file in v:
		print key
		print file 
		a =src_dir+file
		b = dst_dir +key+'/'+file.split('/')[-1]
		print a
		print b
		copyfile(a, b)	

print src_dir
print dst_dir








# output_counter = dict()
# output_Url_list = dict()
# output_counter['miscellaneous'] =0
# fd = 0
# nf= 0
# input_file = 'Genrated_file.csv'
# url_List = []
# with open(input_file,'rb') as f:
# 	rdr = csv.DictReader(f)
# 	for row in rdr:
# 		# print type(row['first name'])
# 		# print row['first name']

# 		for dict_element in ast.literal_eval(row['images']):
# 			url_List.append(dict_element['url'])
# 	for row in url_List:
# 			# print row[0]
# 		parsed_name = parse_Url(row)
# 			# print parsed_name
# 			# print ("parsed name type ",parsed_name)
# 		flag,key =search(input_Dict,parsed_name)
# 		if flag :
# 			if key in output_counter.keys():
# 				output_counter[key] +=1
# 				fd +=1
# 			else:
# 				output_counter[key] =1

# 		else:
# 			output_counter['miscellaneous'] +=1
# 			nf +=1
# print ("found count ", fd)
# print ("Not Found", nf)		
# print output_counter				

# for category Matrix representation
# csv_path = "data.csv"
# with codecs.open(csv_path, "rb",'utf-8') as f_obj:
# 	list_of_Url =csv_reader(f_obj)
# 	counter = 0
# 	print list_of_Url
# 	print type(list_of_Url)
# 	# print len(list_of_Url)
# 	for row in list_of_Url:
# 		# print row[0]
# 		parsed_name = parse_Url(row[0])
# 		# print parsed_name
# 		# print ("parsed name type ",parsed_name)
# 		flag,key =search(input_Dict,parsed_name)
# 		if flag :
# 			if key in output_counter.keys():
# 				output_counter[key] +=1
# 				f +=1
# 			else:
# 				output_counter[key] =1

# 		else:
# 			output_counter['miscellaneous'] +=1
# 			nf +=1
				
		
# 		print flag
# 		print key
# 		print row[0]

# print ("found count ", f)
# print ("Not Found", nf)		
# print output_counter
	


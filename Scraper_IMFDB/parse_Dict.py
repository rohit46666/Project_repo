import csv
import codecs
import re

def csv_reader(file_obj):
	reader = csv.reader(file_obj)
	return reader
	# for row in reader:
	# 	print row[0]
	# 	list_t = parse_Url(row[0])
	# 	# print list_t
	# return list_t	


def search(input_dict, searchList):
	
	for key,k in input_dict.items():
		# print k
		flag = any(elem in searchList  for elem in k)
		if flag :
			break
		# print flag
	return flag,key
        
# csvReader = csv.reader(codecs.open('file.csv', 'rU', 'utf 16'))



def parse_Url(str):
	str_list = str.split("/")
	# print str_list
	temp =  str_list[-1].split(".")[0]
	temp = temp.replace('_','-').lower()
	word_list = re.findall(r"[\w']+", temp)

	# print temp
	return word_list 



# parseStr('www.imfdb.org/images/2/24/VChM1 Mosin 12.jpg')
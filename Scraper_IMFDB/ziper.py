import os
import zipfile
def zipdir(path, ziph):
	# ziph is zipfile handle
	for root, dirs, files in os.walk(path):
		print type(files)
		for file in files:
			ziph.write(os.path.join(root, file))

if __name__ == '__main__':
	zipf = zipfile.ZipFile('DataSet.zip', 'w', zipfile.ZIP_DEFLATED,allowZip64=True)
	zipdir('DataSet/', zipf)
	zipf.close()
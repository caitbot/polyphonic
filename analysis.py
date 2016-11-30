import test
import numpy 
import os

string_labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

def analysis(test_file_location):
	i = 0
	output = test.test() #test_files x 11 matrix 
	for file in os.listdir(test_file_location):
		name, ex = os.path.splitext(test_file_location + '/' + file)
		if (ex == '.txt'):
			f = open(test_file_location + '/' + file, 'r')
			label = convert_string_label_to_vector(f.readline().rstrip())
			print(label)
			i+=1

def convert_string_label_to_vector(string_label):
	vector = [0] *11
	vector[string_labels.index(string_label)] = 1;
	return vector



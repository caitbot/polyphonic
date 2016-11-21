import soundfile as sf 
import os
import numpy as np

#returns input for nn with label
def input():
	sampled = []

	#gets data from wave files
	for instrument in os.listdir('./Training'):
		for song in os.listdir('./Training/' + instrument):
			data, samplerate = sf.read("./Training/" + instrument + "/" + song)
			x = Sample(instrument, data)
			sampled.append(x);

	#makes sure all the songs have the same samples
	minimum = sampled[0].data.size/2

	#find minimum samples
	for sample in sampled:
		size = sample.data.size/2
		if size<minimum:
			minimum = size

	for sample in sampled:
		if sample.data.size/2> minimum:
			sample.data = resize(sample.data, minimum)

	return sampled

#resizes numpy array to desired size
def resize(data, size):
	data_size = data.size/2
	difference = data_size - size
	remove = data_size/difference
	for i in range(0, data_size):
		if i%remove == 0:
			np.delete(data, (i), axis = 0)
	return data 



#one sampel of music
class Sample:
	def __init__(self, l, d):
		self.label = l
		self.data = d
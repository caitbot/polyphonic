import soundfile as sf 
import os
import numpy as np

#returns input for nn with label
def input():
	sampled = []

	#gets data from wave files
	for instrument in os.listdir('/mnt/hgfs/link'):
		for song in os.listdir('/mnt/hgfs/link/' + instrument)[:10]:
		    data, samplerate = sf.read("/mnt/hgfs/link/" + instrument + "/" + song)
		    x = Sample(instrument, data)
		    sampled.append(x);

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

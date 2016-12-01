import soundfile as sf 
import os
import numpy as np

current_batch = 0

data_path = '/mnt/hgfs/link/'
insts = sorted(os.listdir(data_path))

#returns input for nn with label
def input():
    samples = []
    labels = []

	#gets data from wave files
    for instrument in insts:
		for song in os.listdir(data_path + instrument)[:10]:
		    data, samplerate = sf.read(data_path + instrument + "/" + song)
		    
		    sampled.append(data);
		    ohv = [0] * 11
		    ohv[insts.index(instrument)] = 1
		    labels.append(ohv)

    return (np.asarray(samples), np.asarray(labels))
    
def get_batch(size):
    global current_batch
    next_batch = current_batch + size
    samples = []
    labels = []

	#gets data from wave files
    for instrument in insts:
		for song in os.listdir(data_path + instrument)[current_batch:next_batch]:
		    data, samplerate = sf.read(data_path + instrument + "/" + song)
		    
		    samples.append(data);
		    ohv = [0] * 11
		    ohv[insts.index(instrument)] = 1
		    labels.append(ohv)
		    
    current_batch = next_batch

    return (samples, np.asarray(labels))


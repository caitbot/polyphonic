import soundfile as sf 
import os
import numpy as np
from random import shuffle

class Sample:
    data = []
    label = []
    
    def __init__(self, data, label, label_name):
        self.data = data
        self.label = label
        self.label_name = label_name

class InputData:
    song_paths = []
    current_batch = 0
    data_path = '/mnt/hgfs/link/'
    insts = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    
    def __init__(self):
        for inst in self.insts:
            for song in os.listdir(self.data_path + inst):
                self.song_paths.append((inst + '/' + song, inst))
                
        shuffle(self.song_paths)
    
    def next_batch(self, size):
        
        next = []
        
        if len(self.song_paths[self.current_batch:]) < size:
        
            next_batch = self.current_batch + len(songs[current_batch:])
            
            for path in self.song_paths[self.current_batch:next_batch]:
                data, samplerate = sf.read(self.data_path + path[0])
                next.append(Sample(data, path[1]))
                
        else:
            next_batch = self.current_batch + size
            
            for path in self.song_paths[self.current_batch:next_batch]:
                data, samplerate = sf.read(self.data_path + path[0])
                ohv = [0] * 11
                ohv[self.insts.index(path[1])] = 1
                next.append(Sample(data, ohv, path[1]))
                
            self.current_batch += size
            
        return next



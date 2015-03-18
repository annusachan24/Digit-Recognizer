import pprint,pickle
import numpy as np


pkl_file = open('train_pickle.pkl_09.npy','rb')

data1=np.load(pkl_file)
pprint.pprint(data1.shape)

#data2=pickle.load(pkl_file)
#pprint.pprint(data2)

pkl_file.close()

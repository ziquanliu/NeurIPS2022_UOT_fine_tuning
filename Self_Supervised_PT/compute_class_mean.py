import pickle
import numpy as np


imagenet_class_count = pickle.load(open('class_mean/caltech/class_count_unnorm.pkl','rb'))

imagenet_class_vec = pickle.load(open('class_mean/caltech/class_vec_unnorm.pkl','rb'))

print(imagenet_class_count)

imagenet_class_mean = np.divide(imagenet_class_vec,np.expand_dims(imagenet_class_count,axis=-1))

pickle.dump(imagenet_class_mean,open('class_mean/caltech/class_mean_unnorm.pkl','wb'))


#print(imagenet_class_count.shape)
#print(imagenet_class_vec.shape)

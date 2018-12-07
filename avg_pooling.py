from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
#model = ResNet50(weights='imagenet', include_top=False, input_shape=(240,320,3))

def extract_features(img_data, model):
    #img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = model.predict(img_data)
    feature=feature.reshape(feature.shape[0], -1)
    return feature
def get_average_pooling(data, model):
    return np.array([extract_features(x, model)  for x in data])


    
    
    

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
import numpy as np
from numpy.linalg import norm

class FeatureExtraction:

    def model_picker(name):
        
        if (name == 'vgg16'):
            model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3),
                          pooling='max')
        elif (name == 'vgg19'):
            model = VGG19(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3),
                          pooling='max')
        elif (name == 'mobilenet'):
            model = MobileNet(weights='imagenet',
                              include_top=False,
                              input_shape=(224, 224, 3),
                              pooling='max',
                              depth_multiplier=1,
                              alpha=1)
        elif (name == 'inception'):
            model = InceptionV3(weights='imagenet',
                                include_top=False,
                                input_shape=(224, 224, 3),
                                pooling='max')
        elif (name == 'resnet'):
            model = ResNet50(weights='imagenet',
                             include_top=False,
                             input_shape=(224, 224, 3),
                            pooling='max')
        elif (name == 'xception'):
            model = Xception(weights='imagenet',
                             include_top=False,
                             input_shape=(224, 224, 3),
                             pooling='max')
        else:
            print("Specified model not available")
        return model
    

    def extract_features(img_path, model):

        input_shape = (224, 224, 3)
        
        img = image.load_img(img_path,
                            target_size=(input_shape[0], input_shape[1]))
        
        img_array = image.img_to_array(img)
        
        expanded_img_array = np.expand_dims(img_array, axis=0)
        
        preprocessed_img = preprocess_input(expanded_img_array)
        
        features = model.predict(preprocessed_img)
        
        flattened_features = features.flatten()
        
        normalized_features = flattened_features / norm(flattened_features)
        
        return normalized_features



# Commented out IPython magic to ensure Python compatibility.
from numpy.linalg import norm
import pickle
from tqdm import tqdm
import os
from Feature_Extraction import FeatureExtraction
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline


#Select A  model
model_architecture = 'resnet'
model = FeatureExtraction.model_picker(model_architecture)



#features = FeatureExtraction.extract_features(r'C:\DataSet\101_ObjectCategories\Leopards\image_0002.jpg', model)
#print(features[0])
#print(len(features))

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
    return file_list

# path to the datasets
root_dir = r'static/dataset/caltech101'
filenames = sorted(get_file_list(root_dir))

feature_list = []

for i in tqdm(range(len(filenames))):
    feature_list.append(FeatureExtraction.extract_features(filenames[i], model))



for i, features in enumerate(feature_list):
    feature_list[i] = features / norm(features)




pickle.dump(filenames, open(r'savedModels/filenames.pickle', 'wb'))
pickle.dump(feature_list,open(r'savedModels/features.pickle', 'wb'))

"""**Classification using KNN**"""

filenames = pickle.load(open(r'savedModels/filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open(r'savedModels/features-caltech101.pickle','rb'))

#k=FeatureExtraction.extract_features(r"C:\Users\User\image search\test\download1.jpg", model)


neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean').fit(feature_list)

random_index = 10
distances, indices = neighbors.kneighbors([feature_list[random_index]])
#distances, indices = neighbors.kneighbors([k])
plt.imshow(mpimg.imread(filenames[random_index]), interpolation='lanczos')
#plt.imshow(mpimg.imread(k), interpolation='lanczos')

plt.imshow(mpimg.imread(filenames[indices[0][0]]), interpolation='lanczos')

plt.imshow(mpimg.imread(filenames[indices[0][1]]), interpolation='lanczos')

plt.imshow(mpimg.imread(filenames[indices[0][2]]), interpolation='lanczos')

plt.imshow(mpimg.imread(filenames[indices[0][3]]), interpolation='lanczos')

plt.imshow(mpimg.imread(filenames[indices[0][4]]), interpolation='lanczos')

for i in range(5):
    print(distances[0][i])






import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from Feature_Extraction import FeatureExtraction
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import urllib.request
UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Importing PIckle Files
filenames=pickle.load(open(r'savedModels/filenames.pickle','rb'),encoding="ASCII")
feature_list=pickle.load(open(r'savedModels/features.pickle','rb'),encoding="ASCII")


# Initial Module for The HTML PAGE	
@app.route('/')
def upload_form():
	return render_template('index.html')

# Module for Uploading the sample image and processing it. 
'''
@app.route('/', methods=['POST'])
def upload_image():
	if 'sample_image' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['sample_image']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed')
		print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		sample_image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
		return render_template('index.html', filename=filename,prediction_text=filename)
        ----------------------------------------------------------------------------------------#
		Below code To be shifted to different route/module, but getting errors
		model_architecture='resnet'
		training_model= FeatureExtraction.model_picker(model_architecture)
		sample_image_features= FeatureExtraction.extract_features(sample_image_path,training_model)
		print(sample_image_features)
		neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean').fit(feature_list)
		random_index = 75
		distances, indices = neighbors.kneighbors([sample_image_features])
		predicted_image= filenames[indices[0][0]]
		return redirect(url_for('predict_neighbours',test_image=sample_image_path))
		return render_template('index.html', result_image_name=predicted_image,prediction_text=filename)

	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)
'''

#Module for uploading and predicting predicting
@app.route('/',methods=['POST'])
def upload_image():
	if 'sample_image' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['sample_image']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed')
		print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		sample_image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
		model_architecture='resnet'
		training_model= FeatureExtraction.model_picker(model_architecture)
		sample_image_features= FeatureExtraction.extract_features(sample_image_path,training_model)
		print(sample_image_features)
		neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean').fit(feature_list)
		distances, indices = neighbors.kneighbors([sample_image_features])

		'''
		x = filenames[indices[0][0]].split("/",2)
		print(x[-1])
		predicted_image= x[-1]
		'''
		predicted_image_list=[]
		for i in range(0,5):
			predicted_image = filenames[indices[0][i]].replace("\\","/")
			predicted_image_list.append(predicted_image)
		#predicted_image= predicted_image.replace()

		return render_template('index.html',uploadedImage=sample_image_path,NearestImageList=predicted_image_list)
       
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)


#Module for retriving Nearest Neighbours
'''
@app.route('/')
def predict_neighbours(test_image):
	print(test_image)
	model_architecture='resnet'
	training_model= FeatureExtraction.model_picker(model_architecture)
	sample_image_features= FeatureExtraction.extract_features(test_image,training_model)
	print(sample_image_features)
	neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean').fit(feature_list)
	#random_index = 75
	distances, indices = neighbors.kneighbors([sample_image_features])
	return render_template('index.html')
'''

# Module for Displaying images

@app.route('/<result_image_name>')
def display_image(result_image_name):
	print('display_image filename: ' + result_image_name)
	return redirect(url_for('static', filename='dataset/' + result_image_name, code=301))



if __name__ == "__main__":
	app.run(debug=True)
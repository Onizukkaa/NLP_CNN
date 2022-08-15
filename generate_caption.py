from pickle import load
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionV3_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# extract features from each photo in the directory
def extract_features_vgg16(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = vgg16_preprocess(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# extract features from each photo in the directory
def extract_features_inceptionV3(filename):
	# load the model
	model = InceptionV3()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	image = load_img(filename, target_size=(299, 299))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = inceptionV3_preprocess(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# extract features from each photo in the directory
def extract_features_resnet(filename):
	# load the model
	model = ResNet50()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = resnet_preprocess(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature


# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text


def get_vgg16_caption(image):
	# load the tokenizer
	tokenizer = load(open('files/tokenizer.pkl', 'rb'))
	# pre-define the max sequence length (from training)
	max_length = 34
	# load the model
	model = load_model('files/vgg16_model-ep003-loss3.637-val_loss3.874.h5')
	# load and prepare the photograph
	photo = extract_features_vgg16(image)
	# generate description
	description = generate_desc(model, tokenizer, photo, max_length)
	description = description.replace("startseq", "")
	description = description.replace("endseq", "")
	return(description)


def get_inceptionV3_caption(image):
	# load the tokenizer
	tokenizer = load(open('files/tokenizer.pkl', 'rb'))
	# pre-define the max sequence length (from training)
	max_length = 34
	# load the model
	model = load_model('files/inceptionV3_model-ep003-loss3.472-val_loss3.755.h5')
	# load and prepare the photograph
	photo = extract_features_inceptionV3(image)
	# generate description
	description = generate_desc(model, tokenizer, photo, max_length)
	description = description.replace("startseq", "")
	description = description.replace("endseq", "")
	return(description)


def get_resnet_caption(image):
	# load the tokenizer
	tokenizer = load(open('files/resnet_tokenizer.pkl', 'rb'))
	# pre-define the max sequence length (from training)
	max_length = 35
	# load the model
	model = load_model('files/resnet_model.h5')
	# load and prepare the photograph
	photo = extract_features_resnet(image)
	# generate description
	description = generate_desc(model, tokenizer, photo, max_length)
	description = description.replace("startseq", "")
	description = description.replace("endseq", "")
	return(description)
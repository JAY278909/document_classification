import io
import json

import numpy as np
from tensorflow.keras.models import load_model
import pickle
import cv2
import os
from keras import backend as k
from PIL import Image ,TiffTags,ExifTags
from io import BytesIO





class Document_Classifier():


	def imgClassification(self,img_bytes):

		try:

			Document_Classification_Model = load_model(os.getcwd() + '/Ml_Models/model_with_unknown.h5')
			Label_Model = pickle.loads(open(os.getcwd() + "/Ml_Models/Label_Binarizer_with_unknown", "rb").read())

			Data = np.array(Image.open(io.BytesIO(img_bytes)))

			Resize_Image = cv2.resize(Data, (128, 128))

			# scale the pixel values to [0, 1]
			Resize_Image = Resize_Image.astype("float") / 255.0

			# check to see if we should flatten the image and add a batch
			# dimension
			Resize_Image = Resize_Image.flatten()
			Reshape_Image = Resize_Image.reshape((1, Resize_Image.shape[0]))

			Prediction = Document_Classification_Model.predict(Reshape_Image.reshape(1, 128, 128, 3))

			# find the class label index with the largest corresponding probability
			Prediction_Numpy = Prediction.argmax(axis=1)[0]
			Label = Label_Model.classes_[Prediction_Numpy]
			k.clear_session()
			return str(Label)

		except Exception as ErrorMessage:

			print(ErrorMessage)
			return str("unknown")
	


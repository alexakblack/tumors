import tensorflow as tf
import tensorflow.keras.models as models

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
model = models.load_model(args["saved_model_path"])
print(model.summary())

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
import cv2
import numpy as np

DATA_SHAPE = (224, 224)


def prep_image(image):
	image = cv2.resize(image, DATA_SHAPE)
	image = np.expand_dims(image, axis=0)
	return image

def predict(prepped_image):
	prediction = list(model.predict(prepped_image)[0])
	predictions_with_labels = list(zip(prediction, CLASSES))
	predictions_with_labels.sort(reverse=True, key=lambda x: x[0])
	
	print("\n\nResults:")
	for confidence, label in predictions_with_labels:
		output = f"{label : >8}: {confidence * 100:.2f}%"
		print(output)

	return predictions_with_labels



if args["input"] == "image":
	image = cv2.imread(args["path"])
	if image is None:
		ap.error("invalid path to image")

	prepped_image = prep_image(image)
	predict(prepped_image)

	cap.release()
	cv2.destroyAllWindows()
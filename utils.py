# utils.py
import cv2
import numpy as np

# List of classes in the same order as model output
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Load image, resize, normalize, and prepare for model prediction
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img

def get_class_labels():
    return CLASS_LABELS

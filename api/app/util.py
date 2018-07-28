import threading
import time
import tensorflow as tf
import base64
import threading
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input


class ModelLoader(threading.Thread):
    """
    Loads pre-trained model
    """

    def __init__(self, model_Path):

        super(ModelLoader, self).__init__()
        self.model = None
        self.InceptionResNetV2_model = None
        self.model_Path = model_Path

    def getInceptionModel(self):
        if self.InceptionResNetV2_model is None:
            return None
        else:
            return self.InceptionResNetV2_model

    def getModel(self):
        if self.model is None:
            return None
        else:
            return self.model

    def loadModel(self):
        """
        Loads model from the model structure and model weights file
        :return: trained model
        """

        print("Model loading started...")
        s = time.clock()
        self.model = load_model(self.model_Path)
        self.InceptionResNetV2_model = InceptionResNetV2(weights='imagenet', include_top=False)
        e = time.clock()
        print("Model is Loaded: {0}; in {1:.2f} seconds".format(self.model, (e - s)))

    def run(self):
        super(ModelLoader, self).run()
        self.loadModel()

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torchvision import datasets, transforms
from core.config import settings
import numpy as np
from PIL import Image
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from io import BytesIO
# import tqdm


class AgricultureRecognition():
    def __init__(self, gpu=False):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
        self.mtcnn = MTCNN(
            thresholds=[0.7, 0.7, 0.8], keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.max_elements = settings.MAX_ELEMENTS
        self.p = None
        self.fruit_md = tf.keras.models.load_model('data_file/fruit_model.h5')
        self.leaf_md = tf.keras.models.load_model('data_file/leaf_model.h5')
        self.bark_md = tf.keras.models.load_model('data_file/bark_model.h5')
        self.flower_md = tf.keras.models.load_model('data_file/flower_model.h5')
        self.trans_data_augmentation = transforms.Compose([
            transforms.Resize((180, 180)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.trans_default = transforms.Compose([
            transforms.Resize((180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.fruit_name = ['Ananas comosus', 'Apple', 'Apricot', 'Avocado', 'Banana', 'Bell Pepper', 'Bitter gourd', 'Black berry', 'Cabbage soub', 'Camellia sinensis', 'Carica Papaya',
                           'Cherry', 'Chickoo', 'Coffea arabica', 'Corn', 'Cucumber', 'Custard apple', 'Daucus carota subsp', 'Durian', 'Eggplant', 'Grape', 'Guava', 'Kiwi', 'Mango', 'Melon', 'Orange', 
                           'Peach', 'Persimmon', 'Plum', 'Potato', 'Pumpkin', 'Rambutan', 'Solanum lycopersicum', 'Soybean', 'Strawberry', 'Turnip', 'Watermelon']
        self.leaf_name = ['Ananas comosus', 'Apple', 'Apricot', 'Asteraceae', 'Avocado', 'Banana', 'Bell pepper', 'Bitter gourd', 'Black berry', 'Blueberry', 'Brassicaceae', 'Cabbage soup', 'Camellia sinensis',
                           'Carica Papaya', 'Cherry', 'Chickoo', 'Coffea arabica', 'Corn', 'Cucumber', 'Custard apple', 'Daisy', 'Durian', 'Eggplant', 'Grape', 'Guava', 'Hydrangea', 'Kiwi', 'Lavandula angustifolia', 
                           'Lily', 'Mango', 'Melon', 'Mint leaves', 'Orange', 'Peach', 'Pepper', 'Persimmon', 'Plum', 'Potato', 'Pumpkin', 'Rambutan', 'Raspberry', 'Rose', 'Solanum lycopersicum', 'Soybean', 'Squash',
                             'Strawberry', 'Sunflower', 'Tomato', 'Tulip', 'Turnip', 'Watermelon']
        self.flower_name = ['Apple', 'Banana', 'Camellia sinensis', 'Chickoo', 'Coffea arabica', 'Cucumber', 'Daisy', 'Dandelion', 'Dill', 'Durian', 'Hydrangea', 'Iridaceae', 'Lavandula angustifolia', 'Lily', 'Mango',
                            'Mint leaves', 'Orange', 'Parsley leaves', 'Peach', 'Persimmon', 'Potato', 'Rose', 'Rosemary leaves', 'Solanum lycopersicum', 'Strawberry', 'Sunflower', 'Tulip']
        self.bark_name = ['Apple', 'Apricot', 'Asteraceae', 'Avocado', 'Banana', 'Cabbage soup', 'Camellia sinensis', 'Carica Papaya', 'Cherry', 'Chickoo', 'Coffea arabica', 'Custard apple',
                          'Daisy', 'Durian', 'Guava', 'Kiwi', 'Lavandula angustifolia', 'Lily', 'Mango', 'Orange', 'Peach', 'Persimmon', 'Plum', 'Potato', 'Rambutan', 'Rose', 'Solanum lycopersicum', 'Strawberry']

    def trans_for_train(self, img):
        return self.trans_data_augmentation(img).unsqueeze(0)

    def trans_for_recognition(self, img):
        return self.trans_default(img).unsqueeze(0)

    def predict(self, image, model, classes_name, key):
        x = tf.keras.preprocessing.image.img_to_array(image)
        test_img = np.expand_dims(x, axis=0)

        prediction = model.predict(test_img)[0]
        pred = np.argmax(prediction)
        percent_pred = -100 * prediction / np.sum(prediction)
        max_percent_pred = sorted(percent_pred, reverse=True)

        result = {
                'key': key,
                'common_name': '{}'.format(classes_name[pred]),
                'percent': '{:.2f}%'.format(max_percent_pred[0])
            }

        return result

    def multi_predict(self, images):
        results = []

        for i in range(len(images)):
            if images[i]['key'] == 'fruit':
                result = self.predict(images[i]['file'],self.fruit_md, self.fruit_name, 'fruit')
                results.append(result)
            elif images[i]['key'] == 'flower':
                result = self.predict(images[i]['file'],self.flower_md, self.flower_name, 'flower')
                results.append(result)
            elif images[i]['key'] == 'leaf':
                result = self.predict(images[i]['file'],self.leaf_md, self.leaf_name, 'leaf')
                results.append(result)
            elif images[i]['key'] == 'bark':
                result = self.predict(images[i]['file'],self.bark_md, self.bark_name, 'bark')
                results.append(result)
        
        return results

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
        self.classes_name = ['Alpinia officinarum', 'Ananas comosus', 'Apple', 'Apple leaf', 'Asteraceae', 'Banana', 'Blueberry leaf', 'Brassicaceae1', 'Brassicaceae2', 'Cabbage soup', 'Camellia sinensis', 'Cherry', 'Cherry leaf', 'Chickoo fruit', 'Coffea arabica', 'Daisy', 'Dandelion', 'Daucus carota subsp', 'Grape leaf', 'Grapes fruit', 'Hydrangea', 'Iridaceae', 'Kiwi fruit', 'Lavandula angustifolia', 'Lily', 'Mango fruit', 'Narcissus tazetta L', 'Orange fruit', 'Peach leaf', 'Pepper leaf', 'Perilla frutescens', 'Piper sarmentosum', 'Potato', 'Potato leaf', 'Raspberry leaf', 'Rose', 'Shiitake mushroom', 'Solanum lycopersicum', 'Soybean leaf', 'Strawberry fruit', 'Strawberry leaf', 'Sunflower', 'Tithonia diversifolia', 'Tomato leaf', 'Tulip', 'Unknow']

    def trans_for_train(self, img):
        return self.trans_data_augmentation(img).unsqueeze(0)

    def trans_for_recognition(self, img):
        return self.trans_default(img).unsqueeze(0)

    def predict(self, image, model):

        results = []

        x = tf.keras.preprocessing.image.img_to_array(image)
        test_img = np.expand_dims(x, axis=0)

        prediction = model.predict(test_img)[0]
        pred = np.argmax(prediction)
        percent_pred = -1000 * prediction / np.sum(prediction)
        max_percent_pred = sorted(percent_pred, reverse=True)

        results.append(
            {
                'common_name': '{}: {:.2f}%'.format(self.classes_name[pred],max_percent_pred[0])
            }
        )

        return results

    def multi_predict(self, images):
        results = []

        for i in range(len(images)):
            if images[i]['key'] == 'fruit':
                result = self.predict(images[i]['file'],self.fruit_md)
                results.append(result)
            elif images[i]['key'] == 'flower':
                result = self.predict(images[i]['file'],self.flower_md)
                results.append(result)
            elif images[i]['key'] == 'leaf':
                result = self.predict(images[i]['file'],self.leaf_md)
                results.append(result)
            elif images[i]['key'] == 'bark':
                result = self.predict(images[i]['file'],self.bark_md)
                results.append(result)

        
        return results

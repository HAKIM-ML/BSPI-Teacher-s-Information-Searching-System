import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os

class ImageClassificationModel:
    def __init__(self, base_dir, img_height = 224, img_width = 224, batch_size = 32, validation_split = 0.2):
        self.base_dir = base_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Data generation
        self.train_datagen = ImageDataGenerator(
            rescale = 1. / 255,
            validation_split = self.validation_split,
            horizontal_flip = True,
            zoom_range = 0.2,
            shear_range = 0.2
        )
        
        self.train_genrator = self._get_data_generator(subset = 'training')
        self.validation_generator = self._get_data_generator(subset = 'validation')
        
        self.model = self._build_model()
        
        
        
    def _get_data_generator(self, subset):
        return self.train_datagen.flow_from_directory(
            self.base_dir,
            target_size = (self.img_height, self.img_width),
            batch_size = self.batch_size,
            class_model = 'categorical',
            subset = subset
        )
        
    def _build_model(self):
        base_model = MobileNetV2(input_shape = (self.img_height, self.img_width, 3), include_top = False, weights = 'imagenet')
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation = 'relu'),
            Dense(self.train_genrator.num_classes, activation = 'softmax')
        ])
        
        model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuary'])
        return model
    
    
    def train(self, epochs = 1):
        history = self.model.fit(
            self.train_genrator,
            validation_data = self.validation_generator,
            epochs = epochs
        )
        return history
    
    def save_model(self, file_path):
        self.model.save(file_path)
        
    
    def summary(self):
        self.model.summary()
        





###########################  USAGE ######################

base_dir = 'data'
image_classifier = ImageClassificationModel(base_dir)

# Print the model
image_classifier.summary()

# Train the model
history = image_classifier.train(epochs=1)

# save the model
ImageClassificationModel.save_model('model.h5')
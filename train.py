
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from skimage.io import imread
from skimage.transform import resize
import datetime

""" Input images Height and Width """
H = 480
W = 480

input_shape = (H, W, 3)
n_classes = 4           # class (sky, ground, large rock and smaller rock)

""" High parameters """
batch_size = 8  # Bath size
lr = 1e-4       # learning rate
epochs = 30     # Number of Epoch

# 1
# function to load data and train test split
def load_data(IMG_DIR, MASK_DIR):
    X, y = process_data(IMG_DIR, MASK_DIR)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 2
# function to return list of image paths and mask paths
def process_data(IMG_DIR, MASK_DIR):
    images = [os.path.join(IMG_DIR, x) for x in sorted(os.listdir(IMG_DIR))]
    masks = [os.path.join(MASK_DIR, x) for x in sorted(os.listdir(MASK_DIR))]
    return images, masks

# 3
# function for tensorflow dataset pipeline
def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

# 4
# function to read image and mask and create one hot encoding for mask
def preprocess(x, y):
    def f(x, y):
        # Convert x and y to python strings
        x = x.decode()
        y = y.decode()
        # Call read_image and read_mask functions
        image = read_image(x)
        mask = read_mask(y)
        return image, mask
    
# 5
# function to read image
def read_image(x):
    """ Read an image, resize, and normalize the pixel values """
    x = cv2.imread(x, cv2.IMREAD_COLOR) # Read in BGR (array) (unit8)
    x = cv2.resize(x, (W, H))           # Resize to (480, 480)
    x = x / 255.0                       # Normalize [0, 225] to [0, 1]
    x = x.astype(np.float32)            # Convert unit8 to float32
    return x

# 6
# function to read mask
def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    # Normalize the mask 0~3 for one-hot encoding
    normalized = np.zeros_like(x, dtype=np.int32)
    normalized[x == 0] = 0
    normalized[x == 29] = 1
    normalized[x == 76] = 2
    normalized[x == 149] = 3
    x = cv2.resize(normalized, (W, H), interpolation=cv2.INTER_NEAREST)
    x = x.astype(np.int32)
    return x

    # Out put is image(float32) and maxk(int32)
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    # Define the mask with 0 and 1
    mask = tf.one_hot(mask, 4, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 4])
    return image, mask


RENDER_IMAGE_DIR_PATH = os.path.expanduser("~/Lunar_Surface_Semantic_Segmentation/archive/images/render") # image
GROUND_MASK_DIR_PATH = os.path.expanduser("~/Lunar_Surface_Semantic_Segmentation/archive/images/clean") # mask

X_train, X_test, y_train, y_test = load_data(RENDER_IMAGE_DIR_PATH, GROUND_MASK_DIR_PATH)
print(f"Dataset:\n Train: {len(X_train)} \n Test: {len(X_test)}")

# calling tf_dataset
train_dataset = tf_dataset(X_train, y_train, batch=batch_size)
valid_dataset = tf_dataset(X_test, y_test, batch=batch_size)


""" Convolution Block (Called in decoder) """
def conv_block(input_tensor, num_filters):
    
    x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor) # Convolution
    x = layers.BatchNormalization()(x)                              # BatchNormal
    x = layers.Activation("relu")(x)                                # Relu Function

    x = layers.Conv2D(num_filters, 3, padding="same")(x) # Convolution
    x = layers.BatchNormalization()(x)                   # BatchNormal
    x = layers.Activation("relu")(x)                     # Relu Activation

    return x

""" Decode Block """
def decoder(input_tensor, skip_connection, num_filters):
    x = layers.UpSampling2D((2, 2))(input_tensor) # Upsampling
    if skip_connection is not None:
        x = layers.Concatenate()([x, skip_connection]) # Skip-Connection

    # Call Convolution block
    x = conv_block(x, num_filters)
    
    return x

""" Creat u-net architecuture with vgg16 as an encoder """
def create_unet(input_shape, n_classes):

    """
    Using vgg16 as encoder
    1. The top layer won't be used since it is connected decoder
    2. Freeze the base to use pre-trained weight
    """

    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # Fleeze the layers weight to use pre-trained weight
    vgg16.trainable = False 

    # Take layers for skip connection
    s1 = vgg16.get_layer("block1_conv2").output  # (480, 480, 64)
    s2 = vgg16.get_layer("block2_conv2").output  # (240, 240, 128)
    s3 = vgg16.get_layer("block3_conv3").output  # (120, 120, 256)
    s4 = vgg16.get_layer("block4_conv3").output  # (60, 60, 512)
    s5 = vgg16.get_layer("block5_conv3").output  # (30, 30, 512)

    # 
    b0 = layers.MaxPooling2D((2, 2), name="block5_pool")(s5)  # (15, 15, 512)

    # 
    b1 = conv_block(b0, 512)
    

    # Decoder
    d0 = decoder(b1, s5, 256)  # decoder0: (30, 30, 256)
    d1 = decoder(d0, s4, 128)  # decoder1: (60, 60, 128)
    d2 = decoder(d1, s3, 64)   # decoder2: (120, 120, 64)
    d3 = decoder(d2, s2, 32)   # decoder: (240, 240, 32)
    d4 = decoder(d3, None, 16) # decoder: (480, 480, 16)

    # Output layer
    outputs = layers.Conv2D(n_classes, 1, padding="same", activation=None, name="final_conv")(d4)
    outputs = layers.Activation("softmax", name="softmax")(outputs)

    
    model = Model(inputs=vgg16.input, outputs=outputs, name="model_3")
    return model

# Call create_unet function
model = create_unet(input_shape, n_classes)
# Display the model architecture
model.summary()

""" Compile the model with categorical crossentropy function and to optimizem, the Adam is utilieze"""
model.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False, name='Adam'),
                    metrics=['accuracy'])

train_steps = len(X_train) // batch_size
valid_steps = len(X_test) // batch_size

current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

"""
ModelCheckpoint   : Save the best model to specific filepath
ReduceLROnPlateau : The callback monitors a quantity and if no imporovement is seen for a 'patience' number of epochs, the learming rate is reduced.
EarlyStopping     : Stop training when a monitored metric has stopped improving
"""
callbacks = [
    ModelCheckpoint(filepath=f'models/LunarModel_2.h5', monitor='val_loss', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss",factor=0.2, patience=5, verbose=1, min_lr=1e-6), # new_lr = lr * factor 
    EarlyStopping(monitor="val_loss", patience=7, verbose=1),
]

history = model.fit(train_dataset,
                         epochs=epochs,
                         steps_per_epoch=train_steps,
                         validation_data=valid_dataset,
                         validation_steps=valid_steps,
                         callbacks=callbacks)

def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid()

    plt.show()

# plot
plot_history(history)
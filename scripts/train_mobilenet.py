
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from skimage.io import imread
from skimage.transform import resize
import datetime

""" Input images Height and Width """
H = 480
W = 480

input_shape = (H, W, 3)
n_classes = 6           # class (sky, ground, large rock and smaller rock)

""" High parameters """
batch_size = 8  # Bath size
lr = 1e-5       # learning rate
epochs = 50     # Number of Epoch

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

    # Out put is image(float32) and maxk(int32)
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    # Define the mask with 0 and 1
    mask = tf.one_hot(mask, n_classes, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, n_classes])
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
    normalized[x == 33] = 1
    normalized[x == 71] = 2
    normalized[x == 117] = 3
    normalized[x == 162] = 4
    normalized[x == 177] = 5
    x = cv2.resize(normalized, (W, H), interpolation=cv2.INTER_NEAREST)
    x = x.astype(np.int32)
    return x

    

# Dataset directory
IMG_DIR = os.path.expanduser("~/ORBSLAM_Semantic_Mapping/LunarAutonomyChallenge/archive/raw")
MASK_DIR = os.path.expanduser("~/ORBSLAM_Semantic_Mapping/LunarAutonomyChallenge/archive/semantic")

X_train, X_test, y_train, y_test = load_data(IMG_DIR, MASK_DIR)
print(f"Dataset:\n Train: {len(X_train)} \n Test: {len(X_test)}")

# calling tf_dataset
train_dataset = tf_dataset(X_train, y_train, batch=batch_size)
valid_dataset = tf_dataset(X_test, y_test, batch=batch_size)


def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def decoder(input_tensor, skip_connection, num_filters):
    x = layers.UpSampling2D((2, 2))(input_tensor)
    if skip_connection is not None:
        # Ensure shapes match for concatenation
        if x.shape[1] > skip_connection.shape[1]:
            x = layers.Cropping2D(cropping=((0, x.shape[1] - skip_connection.shape[1]),
                                           (0, x.shape[2] - skip_connection.shape[2])))(x)
        elif x.shape[1] < skip_connection.shape[1]:
            x = layers.ZeroPadding2D(padding=((0, skip_connection.shape[1] - x.shape[1]),
                                              (0, skip_connection.shape[2] - x.shape[2])))(x)
        x = layers.Concatenate()([x, skip_connection])
    x = conv_block(x, num_filters)
    return x

def create_unet(input_shape, n_classes):
    """
    Using MobileNetV2 as encoder
    1. The top layer won't be used since it is connected to decoder
    2. Freeze the base to use pre-trained weights
    """
    mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Freeze the layers, unfreeze later layers
    for layer in mobilenet.layers:
        if 'block_13' in layer.name or 'block_14' in layer.name or 'block_15' in layer.name or 'block_16' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    # Take layers for skip connections
    s1 = mobilenet.get_layer("block_1_expand_relu").output   # (240, 240, 96)
    s2 = mobilenet.get_layer("block_3_expand_relu").output   # (120, 120, 144)
    s3 = mobilenet.get_layer("block_6_expand_relu").output   # (60, 60, 192)
    s4 = mobilenet.get_layer("block_13_expand_relu").output  # (30, 30, 576)
    s5 = mobilenet.get_layer("block_16_project").output      # (15, 15, 320)

    # Bottleneck
    b0 = layers.MaxPooling2D((2, 2), name="block_final_pool")(s5)  # (7, 7, 320)
    b1 = conv_block(b0, 512)  # (7, 7, 512)

    # Decoder
    d0 = decoder(b1, s5, 256)  # (7, 7) → (14, 14, 256)
    d1 = decoder(d0, s4, 128)  # (14, 14) → (30, 30, 128)
    d2 = decoder(d1, s3, 64)   # (30, 30) → (60, 60, 64)
    d3 = decoder(d2, s2, 32)   # (60, 60) → (120, 120, 32)
    d4 = decoder(d3, s1, 16)   # (120, 120) → (240, 240, 16)

    # Additional upsampling to reach 480x480
    d5 = layers.UpSampling2D((2, 2))(d4)  # (240, 240) → (480, 480, 16)

    # Output layer
    outputs = layers.Conv2D(n_classes, 1, padding="same", activation=None, name="final_conv")(d5)
    outputs = layers.Activation("softmax", name="softmax")(outputs)

    # Create the model
    model = Model(inputs=mobilenet.input, outputs=outputs, name="unet_mobilenetv2")
    return model
    
# Call create_unet function
model = create_unet(input_shape, n_classes)
# Display the model architecture
#model.summary()

""" Compile the model with categorical crossentropy function and to optimizem, the Adam is utiliezed"""
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
    ModelCheckpoint(filepath=f'model/mobilenet_1.h5', monitor='val_loss', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss",factor=0.2, patience=5, verbose=1, min_lr=1e-6), # new_lr = lr * factor 
    EarlyStopping(monitor="val_loss", patience=7, verbose=1),
]

history = model.fit(train_dataset,
                         epochs=epochs,
                         steps_per_epoch=train_steps,
                         validation_data=valid_dataset,
                         validation_steps=valid_steps,
                         callbacks=callbacks)

# Plot train detail
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

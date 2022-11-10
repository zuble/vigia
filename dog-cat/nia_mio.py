import os, shutil, pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.utils import image_dataset_from_directory


import matplotlib.pyplot as plt
import numpy as np


orig_dir = pathlib.Path("dogs-vs-cats/train")
new_base_dir = pathlib.Path("dogs-vs-cats-small")

def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=orig_dir / fname,
                            dst=dir / fname)

#make_subset("train", start_index=0, end_index=1000)
#make_subset("validation", start_index=1000, end_index=1500)
#make_subset("test", start_index=1500, end_index=2500)

train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32)

validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32)

test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)

for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break



'''
WITHOUT DATA AUGMENTATION
'''
def form_model():
    inputs = keras.Input(shape=(180, 180, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])
    return model

def train_model(model):
    callbacks = [keras.callbacks.ModelCheckpoint(
                    filepath="convnet_from_scratch.keras",
                    save_best_only=True,
                    monitor="val_loss")]
    history = model.fit(
                train_dataset,
                epochs=30,
                validation_data=validation_dataset,
                callbacks=callbacks)
    return history

def test_model():
    test_model = keras.models.load_model("convnet_from_scratch.keras")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy wo/ augm: {test_acc:.3f}")

def plt_model(history):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


model = form_model()
#history = train_model(model)
#plt_model(history)
#test_model()



'''
test image
'''
img_path = '/media/jtstudents/HDD/.zuble/vigia/dog-cat/cat.1700.jpg'
img = tf.keras.utils.load_img(img_path, target_size=(180,180))
img_tensor = tf.keras.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
plt.imshow(img_tensor[0])
#plt.show()




'''
model(maps the ins to outs)
    takes an image as input
    and outputs all layers output values (activations) of original model aka feature maps 
'''
#layer_outputs = [layer.output for layer in model.layers[2:11]]
#activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
#activations = activation_model.predict(img_tensor) #one array per layer activation

'''
test feature map output of a layer (it's activation)
'''
#frist_layer = activations[0]             
#print("\nactivations len :",len(activations),"\nfrist layer shape : ",frist_layer.shape)
#plt.matshow(frist_layer[0, :, :, 2], cmap='viridis')
#plt.show()

'''
vizualize all layers activations
'''
def vizualize_layers_outputs():
    layer_names = []
    for layer in model.layers[2:11]:
        layer_names.append(layer.name)
    print("layer names\n",layer_names)
        
    img_per_row = 16

    for layer_names, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1] #featuremap.shape(I,size,seize,n_features)
        n_cols = n_features // img_per_row
        
        display_grid = np.zeros((size*n_cols, img_per_row*size))
        
        for col in range(n_cols):
            for row in range(img_per_row):
                channel_image = layer_activation[0, :, :, col * img_per_row + row]
                channel_image -= channel_image.mean()
                if(channel_image.std()==0):
                    channel_image /= channel_image.std() + 1e-06
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col+1) * size, 
                             row * size : (row+1) * size] = channel_image   

        sclae = 1. / size
        plt.figure(figsize = (sclae * display_grid.shape[1],
                            sclae * display_grid.shape[0]))
        plt.title(layer_names)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.show()    
    
#vizualize_layers_outputs()



'''
vizualize the visual pattern that each filter is meant to respond
by aplying gradient descent to the input image in order to maximize the filter response, 
starting from a blank image

each layer in a convolution network learns a collection of filters
such that their inputs can be expressed as a combination of filters
'''

def layers_w_conv():
    layers_conv = []
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D)):
            layers_conv.append(layer.name)
    print("\nconv layers :",layers_conv)
    return layers_conv

layer_index = 4

layers_conv = layers_w_conv()
layer = model.get_layer(name=layers_conv[layer_index])
print("\nlayers_conv[",layer_index,"]",layers_conv[layer_index],layer.output_shape)

feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)
    

def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


'''
Loss maximization via stochastic gradient ascent
'''
@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return image

img_width = 180
img_height = 180


'''
Function to generate filter visualizations
'''
def generate_filter_pattern(filter_index):
    iterations = 30
    learning_rate = 10.
    image = tf.random.uniform(
        minval=0.4,
        maxval=0.6,
        shape=(1, img_width, img_height, 3))
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0].numpy()


'''
Utility function to convert a tensor into a valid image
'''
def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]
    return image

plt.axis("off")
plt.imshow(deprocess_image(generate_filter_pattern(filter_index=2)))


all_images = []
n_filters_layer = layer.output_shape[3]

for filter_index in range(n_filters_layer):
    print(f"Processing filter {filter_index}")
    image = deprocess_image(
        generate_filter_pattern(filter_index)
    )
    all_images.append(image)
    keras.utils.save_img(f"layer_filter_responses/filters_for_layer_{layers_conv[layer_index]}_{filter_index}.png", image)

'''
"""
save all filter responses in one img
"""
margin = 5
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))
print(cropped_width,cropped_height,width,height)
for i in range(n):
    for j in range(n):
        image = all_images[i * n + j]
        stitched_filters[
            (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j : (cropped_height + margin) * j
            + cropped_height,
            :,
        ] = image

keras.utils.save_img(f"filters_for_layer_{layers_conv[0]}.png", stitched_filters)
'''


'''

"""
WITH DATA AUGMENTATION
"""
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

        
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss")
]
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks)

test_model = keras.models.load_model(
    "convnet_from_scratch_with_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy w/ augm: {test_acc:.3f}")

'''
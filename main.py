import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import os

# Function to load image
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))  # Resize image to match the VGG input size
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Function to extract features from specific layers of VGG19
def get_model():
    vgg = VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False

    # Choose the layers to extract features from
    content_layers = ['block4_conv2']  # Content layer
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']  # Style layers

    # Define the model
    model_outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    model = tf.keras.models.Model(inputs=vgg.input, outputs=model_outputs)
    return model

# Function to calculate content loss
def calculate_content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

# Function to calculate style loss
def calculate_style_loss(style, target):
    # Compute the Gram matrix of the style/target
    style_gram = gram_matrix(style)
    target_gram = gram_matrix(target)
    return tf.reduce_mean(tf.square(style_gram - target_gram))

# Function to calculate total variation loss (used for regularization)
def total_variation_loss(image):
    return tf.reduce_mean(tf.image.total_variation(image))

# Function to compute the Gram matrix
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(a, a, transpose_a=True)
    return gram

# Function to perform the style transfer training
def train(content_image, style_image, model, num_iterations=200, content_weight=1e3, style_weight=1e-2, tv_weight=1e-6):
    # Initialize the target image as the content image
    target_image = tf.Variable(content_image, dtype=tf.float32)

    # Extract features from content and style images
    model_outputs = model(content_image)
    content_features = model_outputs[:1]  # The first output corresponds to content features
    style_features = model_outputs[1:]    # The rest correspond to style features

    # Extract features from the target image
    target_model_outputs = model(target_image)
    target_content_features = target_model_outputs[:1]  # Target content features
    target_style_features = target_model_outputs[1:]    # Target style features

    # Start the training loop
    for iteration in range(num_iterations):
        with tf.GradientTape() as tape:
            # Extract features from the target image
            target_model_outputs = model(target_image)
            target_content = target_model_outputs[:1]
            target_style = target_model_outputs[1:]

            # Compute content loss
            content_loss = calculate_content_loss(target_content[0], content_features[0])

            # Compute style loss
            style_loss = 0
            for i in range(len(target_style)):
                style_loss += calculate_style_loss(target_style[i], style_features[i])
            style_loss /= len(target_style)

            # Compute total variation loss
            tv_loss = total_variation_loss(target_image)

            # Compute total loss
            total_loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss

        # Compute gradients and update the target image
        grads = tape.gradient(total_loss, target_image)
        optimizer.apply_gradients([(grads, target_image)])

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Total loss: {total_loss.numpy()}")

    # Save the generated image after training
    save_image(target_image.numpy(), "output/outputimage.jpg")

    return target_image

# Function to display the generated image
def display_image(img):
    img = np.array(img)
    img = np.squeeze(img, axis=0)
    img = np.clip(img, 0, 255).astype('uint8')
    plt.imshow(img)
    plt.show()

# Function to save the image
def save_image(img, path):
    img = np.array(img)
    img = np.squeeze(img, axis=0)
    img = np.clip(img, 0, 255).astype('uint8')
    Image.fromarray(img).save(path)
    print(f"Image saved at {path}")

# Load content and style images
content_image = load_image('imgs/content.jpg')
style_image = load_image('imgs/style.jpg')

# Prepare the model
model = get_model()

# Set optimizer
optimizer = tf.optimizers.Adam(learning_rate=5e-3)

# Create output folder if it doesn't exist
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Train the model and get the generated image
generated_image = train(content_image, style_image, model)

# Display the generated image
display_image(generated_image)

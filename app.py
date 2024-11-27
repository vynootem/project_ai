import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG19
import numpy as np
from PIL import Image
import os

# Function to load and preprocess an image
def preprocess_image(uploaded_file):
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((256, 256))  # Resize image to match the VGG19 input size
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Function to save the styled image
def save_image(img, path):
    img = np.array(img)
    img = np.squeeze(img, axis=0)
    img = np.clip(img, 0, 255).astype('uint8')
    Image.fromarray(img).save(path)

# Function to calculate content loss
def calculate_content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

# Function to calculate style loss
def calculate_style_loss(style, target):
    style_gram = gram_matrix(style)
    target_gram = gram_matrix(target)
    return tf.reduce_mean(tf.square(style_gram - target_gram))

# Function to calculate total variation loss
def total_variation_loss(image):
    return tf.reduce_mean(tf.image.total_variation(image))

# Function to compute Gram matrix
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(a, a, transpose_a=True)
    return gram

# Function to perform style transfer
def train(content_image, style_image, model, num_iterations=100, content_weight=1e3, style_weight=1e-2, tv_weight=1e-6):
    target_image = tf.Variable(content_image, dtype=tf.float32)

    content_features = model(content_image)[:1]  # Extract content features
    style_features = model(style_image)[1:]  # Extract style features

    optimizer = tf.optimizers.Adam(learning_rate=5e-3)

    for iteration in range(num_iterations):
        with tf.GradientTape() as tape:
            target_model_outputs = model(target_image)
            target_content = target_model_outputs[:1]
            target_style = target_model_outputs[1:]

            content_loss = calculate_content_loss(target_content[0], content_features[0])

            style_loss = 0
            for i in range(len(target_style)):
                style_loss += calculate_style_loss(target_style[i], style_features[i])
            style_loss /= len(target_style)

            tv_loss = total_variation_loss(target_image)

            total_loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss

        grads = tape.gradient(total_loss, target_image)
        optimizer.apply_gradients([(grads, target_image)])
    
    return target_image

# Streamlit App
st.title("Neural Style Transfer Web App")
st.write("Upload a content image and a style image to generate the stylized image.")

# File uploader for content and style images
content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_image_file and style_image_file:
    # Preprocess uploaded images
    content_image = preprocess_image(content_image_file)
    style_image = preprocess_image(style_image_file)

    # Display uploaded images
    st.image(Image.open(content_image_file), caption="Content Image", use_column_width=True)
    st.image(Image.open(style_image_file), caption="Style Image", use_column_width=True)

    # Load VGG19 model
    vgg = VGG19(weights="imagenet", include_top=False)
    content_layers = ["block4_conv2"]
    style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    model_outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    model = tf.keras.models.Model(inputs=vgg.input, outputs=model_outputs)
    model.trainable = False

    # Generate Stylized Image
    st.write("Generating the stylized image... This may take a few minutes.")
    generated_image = train(content_image, style_image, model)

    # Display and Save Output
    st.write("Generated Image:")
    output_path = "output/stylized_image.jpg"
    save_image(generated_image.numpy(), output_path)
    st.image(output_path, caption="Stylized Image", use_column_width=True)

import tensorflow as tf
import numpy as np
import cv2

# =========================
# Grad-CAM Heatmap Function
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x

    if conv_output is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found.")

    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap /= max_val

    return heatmap.numpy()


# =========================
# Overlay Function
# =========================
def overlay_gradcam(image, heatmap):
    heatmap = cv2.resize(heatmap, (64,64))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + image * 255
    superimposed = superimposed / 255

    return superimposed


# =========================
# Full Pipeline Function
# =========================
def generate_gradcam_output(model, image, layer_name):
    img_array = np.expand_dims(image, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, layer_name)

    result = overlay_gradcam(image, heatmap)

    return result
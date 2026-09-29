"""
Plant Health Check — Proposed Modified CNN (PM-CNN)
Implementation of Proposed Modified CNN with Squeeze-and-Excitation (SE) Channel Attention,
Batch Normalization, Residual Connections, and Global Average Pooling for 39-Class Plant Pathology Classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Sequential


def squeeze_and_excitation_block(input_tensor, ratio=16):
    """
    Squeeze-and-Excitation (SE) Channel Attention Module.
    Dynamically recalibrates feature map channel weights based on global spatial context.
    """
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    
    # Excitation: Bottleneck Dense Layers
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    # Scale: Channel-wise Multiplication
    x = layers.Multiply()([input_tensor, se])
    return x


def build_proposed_pm_cnn(input_shape=(224, 224, 3), num_classes=39, include_se=True, include_bn=True):
    """
    Builds the Proposed Modified CNN (PM-CNN) model architecture.
    
    Args:
        input_shape: tuple of (height, width, channels)
        num_classes: number of target output classes (default 39)
        include_se: bool, whether to include Squeeze-and-Excitation attention
        include_bn: bool, whether to include Batch Normalization
        
    Returns:
        model: tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="input_leaf_image")
    
    # Stem Convolution
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, name="stem_conv")(inputs)
    if include_bn:
        x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation('relu', name="stem_relu")(x)
    
    # Stage 1 Block (64 Filters)
    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name="block1_conv")(x)
    if include_bn:
        x = layers.BatchNormalization(name="block1_bn")(x)
    x = layers.Activation('relu', name="block1_relu")(x)
    if include_se:
        x = squeeze_and_excitation_block(x, ratio=8)
    x = layers.MaxPooling2D((2, 2), name="block1_pool")(x)
    
    # Stage 2 Block (128 Filters with Residual Connection)
    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', name="res_shortcut_128")(x)
    
    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name="block2_conv1")(x)
    if include_bn:
        x = layers.BatchNormalization(name="block2_bn1")(x)
    x = layers.Activation('relu', name="block2_relu1")(x)
    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name="block2_conv2")(x)
    if include_bn:
        x = layers.BatchNormalization(name="block2_bn2")(x)
    if include_se:
        x = squeeze_and_excitation_block(x, ratio=16)
    x = layers.MaxPooling2D((2, 2), name="block2_pool")(x)
    
    # Add Residual Connection
    x = layers.Add(name="block2_add")([x, residual])
    x = layers.Activation('relu', name="block2_out_relu")(x)
    
    # Stage 3 Block (256 Filters)
    x = layers.Conv2D(256, (3, 3), padding='same', use_bias=False, name="block3_conv")(x)
    if include_bn:
        x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.Activation('relu', name="block3_relu")(x)
    if include_se:
        x = squeeze_and_excitation_block(x, ratio=16)
    x = layers.MaxPooling2D((2, 2), name="block3_pool")(x)
    
    # Stage 4 Bottleneck Block (512 Filters)
    x = layers.Conv2D(512, (3, 3), padding='same', use_bias=False, name="block4_conv")(x)
    if include_bn:
        x = layers.BatchNormalization(name="block4_bn")(x)
    x = layers.Activation('relu', name="block4_relu")(x)
    if include_se:
        x = squeeze_and_excitation_block(x, ratio=16)
    x = layers.SpatialDropout2D(0.30, name="block4_spatial_dropout")(x)
    
    # Global Average Pooling Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Dense Classification Layer
    x = layers.Dense(256, use_bias=False, name="dense_features")(x)
    if include_bn:
        x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Activation('relu', name="dense_relu")(x)
    x = layers.Dropout(0.40, name="dense_dropout")(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Proposed_PM_CNN")
    return model


def get_pm_cnn_summary():
    """
    Returns parameter complexity metadata for Proposed PM-CNN.
    """
    model = build_proposed_pm_cnn()
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        "name": "Proposed PM-CNN (ResSE-CNN)",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "input_shape": (224, 224, 3),
        "num_classes": 39,
        "description": "Custom Deep Residual Architecture with Squeeze-and-Excitation Channel Attention and Global Average Pooling."
    }


if __name__ == "__main__":
    pm_model = build_proposed_pm_cnn()
    pm_model.summary()
    info = get_pm_cnn_summary()
    print(f"\n[OK] PM-CNN Built Successfully. Total Parameters: {info['total_params']:,}")

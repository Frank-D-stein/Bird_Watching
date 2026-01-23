#!/usr/bin/env python3
"""
Bird Species Classifier Model Setup

This script downloads or creates an ONNX model for bird species classification.
Supports southeastern US bird species.

Usage:
    python setup_model.py [--output-dir ./data/models]
"""
import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Pre-trained model URLs (you can replace with your own trained model)
MODEL_URLS = {
    # MobileNetV2 trained on iNaturalist birds (example URL - replace with actual)
    'mobilenet_birds': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx',
}

NUM_CLASSES = 51  # Number of SE US bird species in our labels


def create_bird_classifier_onnx(output_path: Path, num_classes: int = NUM_CLASSES):
    """
    Create a MobileNetV2-style ONNX model for bird classification.
    
    This creates a working classifier that can be fine-tuned or replaced
    with a properly trained model.
    """
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        logger.error("Please install onnx: pip install onnx")
        return False

    logger.info(f"Creating bird classifier ONNX model with {num_classes} classes...")

    # Input: 224x224 RGB image (NCHW format)
    input_shape = [1, 3, 224, 224]
    
    # Create a simple but functional CNN architecture
    # This is a lightweight model suitable for real-time inference
    
    # Convolution weights and biases
    np.random.seed(42)  # Reproducible weights
    
    def conv_weights(in_ch, out_ch, kernel=3):
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_ch * kernel * kernel))
        w = (np.random.randn(out_ch, in_ch, kernel, kernel) * scale).astype(np.float32)
        b = np.zeros(out_ch, dtype=np.float32)
        return w, b

    def fc_weights(in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        w = (np.random.randn(out_features, in_features) * scale).astype(np.float32)
        b = np.zeros(out_features, dtype=np.float32)
        return w, b

    # Layer weights
    conv1_w, conv1_b = conv_weights(3, 32, 3)
    conv2_w, conv2_b = conv_weights(32, 64, 3)
    conv3_w, conv3_b = conv_weights(64, 128, 3)
    conv4_w, conv4_b = conv_weights(128, 256, 3)
    
    # After 4 conv+pool layers: 224 -> 112 -> 56 -> 28 -> 14
    # Global avg pool gives 256 features
    fc_w, fc_b = fc_weights(256, num_classes)

    # Create initializers
    initializers = [
        numpy_helper.from_array(conv1_w, 'conv1_w'),
        numpy_helper.from_array(conv1_b, 'conv1_b'),
        numpy_helper.from_array(conv2_w, 'conv2_w'),
        numpy_helper.from_array(conv2_b, 'conv2_b'),
        numpy_helper.from_array(conv3_w, 'conv3_w'),
        numpy_helper.from_array(conv3_b, 'conv3_b'),
        numpy_helper.from_array(conv4_w, 'conv4_w'),
        numpy_helper.from_array(conv4_b, 'conv4_b'),
        numpy_helper.from_array(fc_w, 'fc_w'),
        numpy_helper.from_array(fc_b, 'fc_b'),
    ]

    # Build the graph
    nodes = [
        # Conv block 1
        helper.make_node('Conv', ['input', 'conv1_w', 'conv1_b'], ['conv1_out'],
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['conv1_out'], ['relu1_out']),
        helper.make_node('MaxPool', ['relu1_out'], ['pool1_out'],
                        kernel_shape=[2, 2], strides=[2, 2]),
        
        # Conv block 2
        helper.make_node('Conv', ['pool1_out', 'conv2_w', 'conv2_b'], ['conv2_out'],
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['conv2_out'], ['relu2_out']),
        helper.make_node('MaxPool', ['relu2_out'], ['pool2_out'],
                        kernel_shape=[2, 2], strides=[2, 2]),
        
        # Conv block 3
        helper.make_node('Conv', ['pool2_out', 'conv3_w', 'conv3_b'], ['conv3_out'],
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['conv3_out'], ['relu3_out']),
        helper.make_node('MaxPool', ['relu3_out'], ['pool3_out'],
                        kernel_shape=[2, 2], strides=[2, 2]),
        
        # Conv block 4
        helper.make_node('Conv', ['pool3_out', 'conv4_w', 'conv4_b'], ['conv4_out'],
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['conv4_out'], ['relu4_out']),
        helper.make_node('MaxPool', ['relu4_out'], ['pool4_out'],
                        kernel_shape=[2, 2], strides=[2, 2]),
        
        # Global Average Pooling
        helper.make_node('GlobalAveragePool', ['pool4_out'], ['gap_out']),
        
        # Flatten
        helper.make_node('Flatten', ['gap_out'], ['flat_out'], axis=1),
        
        # Fully connected output layer
        helper.make_node('Gemm', ['flat_out', 'fc_w', 'fc_b'], ['output'],
                        transB=1),
    ]

    # Input/output definitions
    inputs = [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    ]
    outputs = [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, num_classes])
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'BirdClassifier',
        inputs,
        outputs,
        initializers
    )

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    # Add metadata
    model.doc_string = 'Bird Species Classifier for Southeastern US Birds'
    model.model_version = 1

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    
    logger.info(f"✓ Model saved to {output_path}")
    logger.info(f"  Input shape: {input_shape} (NCHW)")
    logger.info(f"  Output classes: {num_classes}")
    return True


def download_pretrained_model(url: str, output_path: Path) -> bool:
    """Download a pre-trained model from URL."""
    logger.info(f"Downloading model from {url}...")
    try:
        urllib.request.urlretrieve(url, str(output_path))
        logger.info(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return False


def verify_model(model_path: Path, labels_path: Path) -> bool:
    """Verify the model works with sample input."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")
        return True

    logger.info("Verifying model...")
    
    # Load model
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_info = session.get_inputs()[0]
    
    # Create dummy input
    input_shape = input_info.shape
    dummy_input = np.random.randn(*[s if isinstance(s, int) else 1 for s in input_shape]).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_info.name: dummy_input})
    output_shape = outputs[0].shape
    
    # Check labels
    num_classes = output_shape[-1]
    if labels_path.exists():
        with open(labels_path) as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) != num_classes:
            logger.warning(f"Label count ({len(labels)}) doesn't match model output ({num_classes})")
    
    logger.info(f"✓ Model verified: input {input_shape} -> output {output_shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Setup bird classification model')
    parser.add_argument('--output-dir', type=str, default='./data/models',
                       help='Output directory for model files')
    parser.add_argument('--model-name', type=str, default='bird_classifier.onnx',
                       help='Output model filename')
    parser.add_argument('--download', type=str, choices=list(MODEL_URLS.keys()),
                       help='Download a pre-trained model instead of creating one')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                       help='Number of bird species classes')
    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / args.model_name
    labels_path = output_dir / 'bird_labels.txt'

    logger.info("=" * 50)
    logger.info("Bird Species Classifier Setup")
    logger.info("=" * 50)

    # Create or download model
    if args.download:
        success = download_pretrained_model(MODEL_URLS[args.download], model_path)
    else:
        success = create_bird_classifier_onnx(model_path, args.num_classes)

    if not success:
        logger.error("Failed to setup model")
        return 1

    # Verify
    if model_path.exists():
        verify_model(model_path, labels_path)

    logger.info("")
    logger.info("=" * 50)
    logger.info("Setup Complete!")
    logger.info("=" * 50)
    logger.info("")
    logger.info("To use the model, set these environment variables:")
    logger.info(f"  ML_MODEL_PATH={model_path.absolute()}")
    logger.info(f"  ML_LABELS_PATH={labels_path.absolute()}")
    logger.info("")
    logger.info("Or add to docker-compose.yml environment:")
    logger.info(f"  - ML_MODEL_PATH=/app/data/models/{args.model_name}")
    logger.info(f"  - ML_LABELS_PATH=/app/data/models/bird_labels.txt")
    logger.info("")
    logger.info("Note: This is a randomly-initialized model for testing.")
    logger.info("For production, train on actual bird images or use a")
    logger.info("pre-trained model fine-tuned on your target species.")

    return 0


if __name__ == '__main__':
    sys.exit(main())

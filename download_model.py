#!/usr/bin/env python3
"""
Download and set up a pre-trained bird classification model.

This script provides options to:
1. Download a MobileNetV2 model pre-trained on ImageNet (general objects)
2. Set up for fine-tuning on bird species
3. Use transfer learning with your own bird images

Usage:
    python download_model.py --model mobilenet
    python download_model.py --model efficientnet
"""
import argparse
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append('torch')
    
    try:
        import torchvision
    except ImportError:
        missing.append('torchvision')
    
    try:
        import onnx
    except ImportError:
        missing.append('onnx')
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def download_mobilenet_v2(output_dir, num_classes=51):
    """
    Download MobileNetV2 and export to ONNX.
    
    This model is lightweight and fast, suitable for real-time detection.
    """
    import torch
    import torchvision.models as models
    
    print("Downloading MobileNetV2...")
    
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Replace classifier for bird species
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    
    # Initialize the new classifier with better weights
    torch.nn.init.xavier_uniform_(model.classifier[1].weight)
    torch.nn.init.zeros_(model.classifier[1].bias)
    
    model.eval()
    
    # Export to ONNX
    output_path = Path(output_dir) / 'bird_classifier.onnx'
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"Model saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def download_efficientnet_b0(output_dir, num_classes=51):
    """
    Download EfficientNet-B0 and export to ONNX.
    
    This model has better accuracy than MobileNet but is slightly larger.
    """
    import torch
    import torchvision.models as models
    
    print("Downloading EfficientNet-B0...")
    
    # Load pre-trained EfficientNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace classifier for bird species
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    
    # Initialize the new classifier
    torch.nn.init.xavier_uniform_(model.classifier[1].weight)
    torch.nn.init.zeros_(model.classifier[1].bias)
    
    model.eval()
    
    # Export to ONNX
    output_path = Path(output_dir) / 'bird_classifier.onnx'
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"Model saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def create_training_script(output_dir):
    """Create a script for fine-tuning on bird images."""
    script_path = Path(output_dir) / 'train_bird_model.py'
    
    script_content = '''#!/usr/bin/env python3
"""
Fine-tune a pre-trained model on bird images with CUDA/GPU support.

Features:
- Automatic GPU detection and utilization
- Mixed precision training (AMP) for 2-3x speedup on modern GPUs
- Multi-GPU support with DataParallel
- Progress bars and ETA
- Automatic batch size adjustment for GPU memory

Directory structure expected:
    data/training/
        Northern Cardinal/
            img1.jpg
            img2.jpg
        Blue Jay/
            img1.jpg
            img2.jpg
        ...

Usage:
    python train_bird_model.py --data-dir ./data/training --epochs 10
    python train_bird_model.py --data-dir ./data/training --epochs 20 --batch-size 64 --amp
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from pathlib import Path


def get_device_info():
    """Get detailed device information."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
        'recommended_device': 'cpu'
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['recommended_device'] = 'cuda'
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'index': i,
                'name': props.name,
                'memory_gb': props.total_memory / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}"
            })
    
    return info


def print_device_info(device_info):
    """Print device information."""
    print("\\n" + "="*60)
    print("DEVICE CONFIGURATION")
    print("="*60)
    
    if device_info['cuda_available']:
        print(f"✓ CUDA is available!")
        print(f"  GPU Count: {device_info['device_count']}")
        for dev in device_info['devices']:
            print(f"  [{dev['index']}] {dev['name']}")
            print(f"      Memory: {dev['memory_gb']:.1f} GB")
            print(f"      Compute: {dev['compute_capability']}")
        print(f"\\n  Using: {device_info['recommended_device'].upper()}")
    else:
        print("✗ CUDA not available - using CPU")
        print("  Training will be slower. To enable GPU:")
        print("  1. Install CUDA Toolkit from nvidia.com")
        print("  2. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("="*60 + "\\n")


def get_optimal_workers():
    """Get optimal number of data loader workers."""
    try:
        cpu_count = os.cpu_count() or 4
        # Use fewer workers on Windows due to spawn overhead
        if sys.platform == 'win32':
            return min(4, cpu_count)
        return min(8, cpu_count)
    except:
        return 4


def train_model(data_dir, epochs=10, batch_size=32, learning_rate=0.001, 
                use_amp=True, num_workers=None, model_type='mobilenet'):
    """Train the bird classifier with CUDA support."""
    
    # Get device info
    device_info = get_device_info()
    print_device_info(device_info)
    
    device = torch.device(device_info['recommended_device'])
    use_cuda = device.type == 'cuda'
    
    # Disable AMP on CPU
    if not use_cuda:
        use_amp = False
    
    # Auto-adjust batch size for GPU memory
    if use_cuda and batch_size == 32:
        gpu_mem = device_info['devices'][0]['memory_gb']
        if gpu_mem >= 8:
            batch_size = 64
            print(f"Auto-adjusted batch size to {batch_size} for {gpu_mem:.1f}GB GPU")
        elif gpu_mem < 4:
            batch_size = 16
            print(f"Auto-adjusted batch size to {batch_size} for {gpu_mem:.1f}GB GPU")
    
    # Set workers
    if num_workers is None:
        num_workers = get_optimal_workers()
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Mixed Precision (AMP): {use_amp}")
    print(f"  Model: {model_type}")
    print()
    
    # Data transforms with more augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    
    if not train_dir.exists():
        train_dir = Path(data_dir)
        val_dir = Path(data_dir)
    
    print(f"Loading data from: {train_dir}")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # Pin memory for faster GPU transfer
    pin_memory = use_cuda
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    num_classes = len(train_dataset.classes)
    print(f"\\nFound {num_classes} classes with {len(train_dataset)} training images")
    print(f"Classes: {train_dataset.classes[:5]}..." if num_classes > 5 else f"Classes: {train_dataset.classes}")
    
    # Save class names
    with open('bird_labels.txt', 'w') as f:
        for class_name in train_dataset.classes:
            f.write(f"{class_name}\\n")
    
    # Load pre-trained model
    print(f"\\nLoading {model_type} model...")
    if model_type == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        # Freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
    else:  # mobilenet
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        # Freeze early layers
        for param in model.features[:14].parameters():
            param.requires_grad = False
    
    # Multi-GPU support
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/100)
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\\n")
    
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        batch_count = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            
            # Progress indicator
            if (batch_idx + 1) % max(1, batch_count // 5) == 0:
                progress = (batch_idx + 1) / batch_count * 100
                print(f"  Epoch {epoch+1}: {progress:5.1f}% complete", end="\\r")
        
        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        
        # Validation phase
        model.eval()
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        improved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            # Save best model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), 'best_model.pth')
            improved = " ★ NEW BEST"
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Train: {train_acc:.1%} | Val: {val_acc:.1%} | LR: {current_lr:.2e} | {epoch_time:.1f}s{improved}")
        
    total_time = time.time() - total_start
    
    # Clear GPU cache
    if use_cuda:
        torch.cuda.empty_cache()
    
    print(f"\\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_acc:.1%}")
    
    # Export to ONNX
    print(f"\\nExporting model to ONNX...")
    
    # Load best model
    model_to_export = models.mobilenet_v2() if model_type == 'mobilenet' else models.efficientnet_b0()
    if model_type == 'efficientnet':
        in_features = model_to_export.classifier[1].in_features
        model_to_export.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        model_to_export.classifier[1] = nn.Linear(model_to_export.last_channel, num_classes)
    
    model_to_export.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model_to_export.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model_to_export,
        dummy_input,
        'bird_classifier.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    
    # Get file sizes
    import os
    onnx_size = os.path.getsize('bird_classifier.onnx') / 1024 / 1024
    pth_size = os.path.getsize('best_model.pth') / 1024 / 1024
    
    print(f"\\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print(f"  bird_classifier.onnx  ({onnx_size:.1f} MB) - Deploy this")
    print(f"  bird_labels.txt       (labels)")
    print(f"  best_model.pth        ({pth_size:.1f} MB) - PyTorch checkpoint")
    
    print(f"\\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Copy model to data/models/:")
    print("   cp bird_classifier.onnx ./data/models/")
    print("   cp bird_labels.txt ./data/models/")
    print("2. Rebuild Docker:")
    print("   docker-compose up -d --build")
    print(f"{'='*60}\\n")
    
    return best_acc, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train bird classifier with CUDA/GPU support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (auto-detects GPU)
  python train_bird_model.py --data-dir ./data/training
  
  # Train with larger batch size on GPU
  python train_bird_model.py --data-dir ./data/training --batch-size 64 --epochs 20
  
  # Force CPU training
  CUDA_VISIBLE_DEVICES="" python train_bird_model.py --data-dir ./data/training
  
  # Use EfficientNet (more accurate)
  python train_bird_model.py --data-dir ./data/training --model efficientnet
        """
    )
    parser.add_argument('--data-dir', required=True, help='Directory with training images')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (auto-adjusted for GPU)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--workers', type=int, default=None, help='Data loader workers (auto-detected)')
    parser.add_argument('--model', choices=['mobilenet', 'efficientnet'], default='mobilenet',
                       help='Model architecture (default: mobilenet)')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    args = parser.parse_args()
    
    train_model(
        args.data_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.lr,
        use_amp=not args.no_amp,
        num_workers=args.workers,
        model_type=args.model
    )
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Training script saved to: {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description='Download pre-trained bird classification model')
    parser.add_argument('--model', choices=['mobilenet', 'efficientnet'], default='mobilenet',
                       help='Model architecture to use')
    parser.add_argument('--output-dir', default='./data/models',
                       help='Output directory for the model')
    parser.add_argument('--num-classes', type=int, default=51,
                       help='Number of bird species classes')
    parser.add_argument('--create-training-script', action='store_true',
                       help='Also create a training script for fine-tuning')
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("\nTo use this script, install PyTorch first:")
        print("  pip install torch torchvision onnx")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model
    if args.model == 'mobilenet':
        download_mobilenet_v2(output_dir, args.num_classes)
    elif args.model == 'efficientnet':
        download_efficientnet_b0(output_dir, args.num_classes)
    
    # Create training script if requested
    if args.create_training_script:
        create_training_script(output_dir)
    
    print("\n✓ Model ready!")
    print("\nTo improve accuracy further, collect bird images and run:")
    print(f"  python {output_dir}/train_bird_model.py --data-dir ./data/training --epochs 10")


if __name__ == '__main__':
    main()

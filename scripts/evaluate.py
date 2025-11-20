"""
Evaluation script for MS lesion segmentation
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import SwinUNETR
from src.data import TestDataset, MSLesionDataset
from src.evaluation import dice_score, sensitivity, specificity, compute_all_metrics
from src.evaluation.visualization import save_results
from src.utils.io_utils import load_nifti


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str, config: dict, device: str) -> torch.nn.Module:
    """Load trained model."""
    model_config_path = config['model']['config_path']
    model_config = load_config(model_config_path)
    
    model_params = model_config['model']
    model = SwinUNETR(
        in_channels=model_params['in_channels'],
        out_channels=model_params['out_channels'],
        img_size=tuple(model_params['img_size']),
        feature_size=model_params['feature_size'],
        use_attention=model_params['use_attention'],
        attention_type=model_params.get('attention_type', 'cbam')
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def evaluate_on_dataset(model: torch.nn.Module, data_loader: DataLoader, 
                        device: str, save_predictions: bool = False,
                        output_dir: str = None):
    """Evaluate model on a dataset."""
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch.get('mask')
            
            if masks is not None:
                masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            # Compute metrics if ground truth available
            if masks is not None:
                batch_metrics = []
                for i in range(predictions.shape[0]):
                    pred = predictions[i].cpu().numpy()
                    mask = masks[i].cpu().numpy()
                    metrics = compute_all_metrics(pred, mask)
                    batch_metrics.append(metrics)
                
                all_metrics.extend(batch_metrics)
            
            # Save predictions
            if save_predictions and output_dir:
                predictions_np = predictions.cpu().numpy()
                for i in range(predictions_np.shape[0]):
                    patient_id = batch['patient_id'][i]
                    timepoint = batch['timepoint'][i]
                    save_results(
                        predictions_np[i],
                        output_dir,
                        patient_id,
                        timepoint
                    )
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isinf(m[key])]
            if values:
                avg_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return avg_metrics
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate MS Lesion Segmentation Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save predictions')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction masks as NIfTI files')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("MS Lesion Segmentation Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path, config, device)
    print("Model loaded successfully")
    
    # Determine test directory
    test_dir = args.test_dir or config['data']['data_dir']
    
    # Create test dataset
    print(f"\nLoading test data from: {test_dir}")
    
    # Check if it's test data (no masks) or validation data (with masks)
    try:
        dataset = TestDataset(
            data_dir=test_dir,
            use_preprocessed=config['data']['use_preprocessed'],
            normalize=config['data']['normalize'],
            augmentation=False,
            target_size=tuple(config['data']['target_size']) if config['data'].get('target_size') else None,
            modalities=config['data']['modalities']
        )
        print("Using TestDataset (no ground truth)")
    except:
        # Fall back to regular dataset if test pattern doesn't match
        dataset = MSLesionDataset(
            data_dir=test_dir,
            use_preprocessed=config['data']['use_preprocessed'],
            normalize=config['data']['normalize'],
            augmentation=False,
            target_size=tuple(config['data']['target_size']) if config['data'].get('target_size') else None,
            modalities=config['data']['modalities']
        )
        print("Using MSLesionDataset (with ground truth)")
    
    test_loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"Test samples: {len(dataset)}")
    
    # Create output directory
    output_dir = args.output_dir
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Predictions will be saved to: {output_dir}")
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_on_dataset(
        model,
        test_loader,
        device,
        save_predictions=args.save_predictions,
        output_dir=output_dir
    )
    
    # Print results
    if metrics:
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for metric_name, metric_values in metrics.items():
            print(f"{metric_name}:")
            print(f"  Mean: {metric_values['mean']:.4f} ± {metric_values['std']:.4f}")
            print(f"  Range: [{metric_values['min']:.4f}, {metric_values['max']:.4f}]")
        print("=" * 60)
    else:
        print("\nEvaluation completed. No ground truth available for metrics.")
        if args.save_predictions:
            print(f"Predictions saved to: {output_dir}")


if __name__ == '__main__':
    main()


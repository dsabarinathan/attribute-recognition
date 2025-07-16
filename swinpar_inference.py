"""
SwinPAR: Inference Script for Person Attribute Recognition
A clean and simple interface for running inference with the SwinPAR model
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import model from the main model file
from swinPAR_model import SwinPAR, LABEL_COLUMNS, get_swinpar_transforms


class SwinPARInference:
    """SwinPAR Inference Class"""
    
    def __init__(self, model_path, device='auto'):
        """
        Initialize SwinPAR inference
        
        Args:
            model_path (str): Path to the trained model checkpoint
            device (str): Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.model, self.checkpoint = self._load_model(model_path)
        self.label_columns = self.checkpoint.get('label_columns', LABEL_COLUMNS)
        self.transform = self._get_inference_transform()
        
        print(f"SwinPAR model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self._count_parameters():,}")
        print(f"Number of attributes: {len(self.label_columns)}")
    
    def _setup_device(self, device):
        """Setup device for inference"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path):
        """Load the trained model"""
        print(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = SwinPAR(
            img_size=checkpoint.get('img_size', 224),
            num_classes=checkpoint.get('num_classes', 40),
            use_pretrained_swin=False
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model, checkpoint
    
    def _get_inference_transform(self):
        """Get inference transforms"""
        return A.Compose([
            A.Resize(width=224, height=224, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path
        
        # Apply transforms
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image
    
    def predict_single(self, image_path, threshold=0.5):
        """
        Predict attributes for a single image
        
        Args:
            image_path (str): Path to the image file
            threshold (float): Threshold for binary classification
            
        Returns:
            dict: Prediction results with probabilities and binary predictions
        """
        # Preprocess image
        image_tensor, original_image = self._preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = outputs.cpu().numpy()[0]  # Remove batch dimension
        
        # Convert to predictions
        predictions = (probabilities > threshold).astype(int)
        
        # Create result dictionary
        result = {
            'image_path': str(image_path),
            'probabilities': {name: float(prob) for name, prob in zip(self.label_columns, probabilities)},
            'predictions': {name: int(pred) for name, pred in zip(self.label_columns, predictions)},
            'binary_vector': predictions.tolist(),
            'threshold': threshold,
            'positive_attributes': [name for name, pred in zip(self.label_columns, predictions) if pred == 1],
            'confidence_scores': {name: float(prob) for name, prob in zip(self.label_columns, probabilities) if probabilities[i] > threshold for i, name in enumerate(self.label_columns)}
        }
        
        return result
    
    def predict_batch(self, image_paths, threshold=0.5, batch_size=32):
        """
        Predict attributes for multiple images
        
        Args:
            image_paths (list): List of image paths
            threshold (float): Threshold for binary classification
            batch_size (int): Batch size for inference
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    image_tensor, _ = self._preprocess_image(path)
                    batch_images.append(image_tensor)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Stack batch
            batch_tensor = torch.cat(batch_images, dim=0)
            
            # Run inference
            with torch.no_grad():
                batch_tensor = batch_tensor.to(self.device)
                outputs = self.model(batch_tensor)
                probabilities = outputs.cpu().numpy()
            
            # Process results
            for j, (path, probs) in enumerate(zip(batch_paths, probabilities)):
                predictions = (probs > threshold).astype(int)
                
                result = {
                    'image_path': str(path),
                    'probabilities': {name: float(prob) for name, prob in zip(self.label_columns, probs)},
                    'predictions': {name: int(pred) for name, pred in zip(self.label_columns, predictions)},
                    'binary_vector': predictions.tolist(),
                    'threshold': threshold,
                    'positive_attributes': [name for name, pred in zip(self.label_columns, predictions) if pred == 1]
                }
                
                results.append(result)
        
        return results
    
    def visualize_prediction(self, image_path, prediction_result, save_path=None, figsize=(15, 8)):
        """
        Visualize prediction results
        
        Args:
            image_path (str): Path to the image
            prediction_result (dict): Prediction result from predict_single()
            save_path (str): Path to save the visualization
            figsize (tuple): Figure size
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Show image
        ax1.imshow(image)
        ax1.set_title("Input Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Show predictions
        positive_attrs = prediction_result['positive_attributes']
        if positive_attrs:
            probabilities = [prediction_result['probabilities'][attr] for attr in positive_attrs]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(positive_attrs))
            bars = ax2.barh(y_pos, probabilities, color='skyblue', alpha=0.7)
            
            # Customize bars
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', ha='left', va='center', fontweight='bold')
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([attr.replace('-', ' ') for attr in positive_attrs])
            ax2.set_xlabel('Confidence Score', fontweight='bold')
            ax2.set_title('Detected Attributes', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No attributes detected\n(above threshold)', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax2.set_title('Detected Attributes', fontsize=14, fontweight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def create_summary_report(self, results, save_path=None):
        """
        Create a summary report from multiple predictions
        
        Args:
            results (list): List of prediction results
            save_path (str): Path to save the report
        """
        if not results:
            print("No results to create report")
            return
        
        # Calculate statistics
        total_images = len(results)
        all_predictions = np.array([result['binary_vector'] for result in results])
        attribute_counts = np.sum(all_predictions, axis=0)
        attribute_percentages = (attribute_counts / total_images) * 100
        
        # Create DataFrame for better visualization
        import pandas as pd
        
        summary_df = pd.DataFrame({
            'Attribute': self.label_columns,
            'Count': attribute_counts,
            'Percentage': attribute_percentages
        })
        
        # Sort by percentage
        summary_df = summary_df.sort_values('Percentage', ascending=False)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Bar chart of top attributes
        top_attrs = summary_df.head(15)
        ax1.barh(range(len(top_attrs)), top_attrs['Percentage'], color='skyblue')
        ax1.set_yticks(range(len(top_attrs)))
        ax1.set_yticklabels([attr.replace('-', ' ') for attr in top_attrs['Attribute']])
        ax1.set_xlabel('Percentage of Images (%)')
        ax1.set_title(f'Top 15 Most Common Attributes (n={total_images})', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Heatmap of attribute categories
        categories = {}
        for attr in self.label_columns:
            category = attr.split('-')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(attr)
        
        category_data = []
        for category, attrs in categories.items():
            category_percentages = [summary_df[summary_df['Attribute'] == attr]['Percentage'].iloc[0] for attr in attrs]
            category_data.append(category_percentages)
        
        # Create heatmap
        max_len = max(len(attrs) for attrs in categories.values())
        heatmap_data = []
        labels = []
        
        for category, attrs in categories.items():
            percentages = [summary_df[summary_df['Attribute'] == attr]['Percentage'].iloc[0] for attr in attrs]
            # Pad with zeros if needed
            while len(percentages) < max_len:
                percentages.append(0)
            heatmap_data.append(percentages)
            labels.append(category)
        
        im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.set_title('Attribute Distribution by Category', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Percentage (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary report saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY REPORT")
        print("="*50)
        print(f"Total images processed: {total_images}")
        print(f"Average attributes per image: {np.mean(np.sum(all_predictions, axis=1)):.2f}")
        print(f"Most common attribute: {summary_df.iloc[0]['Attribute']} ({summary_df.iloc[0]['Percentage']:.1f}%)")
        print(f"Least common attribute: {summary_df.iloc[-1]['Attribute']} ({summary_df.iloc[-1]['Percentage']:.1f}%)")
        
        return summary_df
    
    def get_model_info(self):
        """Get information about the loaded model"""
        info = {
            'model_name': 'SwinPAR',
            'total_parameters': self._count_parameters(),
            'device': str(self.device),
            'num_classes': len(self.label_columns),
            'image_size': self.checkpoint.get('img_size', 224),
            'label_columns': self.label_columns
        }
        
        if 'val_loss' in self.checkpoint:
            info['validation_loss'] = self.checkpoint['val_loss']
        if 'mean_accuracy' in self.checkpoint:
            info['mean_accuracy'] = self.checkpoint['mean_accuracy']
        if 'epoch' in self.checkpoint:
            info['trained_epochs'] = self.checkpoint['epoch']
        
        return info


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='SwinPAR Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image_path', type=str, help='Path to input image')
    parser.add_argument('--image_dir', type=str, help='Path to directory containing images')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--save_results', action='store_true', help='Save results to JSON')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize inference
    inference = SwinPARInference(args.model_path, args.device)
    
    # Print model info
    model_info = inference.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    results = []
    
    # Single image inference
    if args.image_path:
        print(f"\nProcessing single image: {args.image_path}")
        result = inference.predict_single(args.image_path, args.threshold)
        results.append(result)
        
        # Print results
        print(f"\nDetected attributes ({len(result['positive_attributes'])} total):")
        for attr in result['positive_attributes']:
            confidence = result['probabilities'][attr]
            print(f"  - {attr}: {confidence:.3f}")
        
        # Visualize if requested
        if args.visualize:
            vis_path = output_dir / f"visualization_{Path(args.image_path).stem}.png"
            inference.visualize_prediction(args.image_path, result, str(vis_path))
    
    # Directory inference
    elif args.image_dir:
        print(f"\nProcessing images from directory: {args.image_dir}")
        
        # Get all image files
        image_dir = Path(args.image_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_paths)} images")
        
        # Run batch inference
        results = inference.predict_batch([str(p) for p in image_paths], args.threshold, args.batch_size)
        
        print(f"Successfully processed {len(results)} images")
        
        # Create summary report
        if args.visualize:
            report_path = output_dir / "summary_report.png"
            summary_df = inference.create_summary_report(results, str(report_path))
    
    else:
        print("Please provide either --image_path or --image_dir")
        return
    
    # Save results to JSON
    if args.save_results and results:
        results_path = output_dir / "inference_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()

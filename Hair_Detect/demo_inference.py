#!/usr/bin/env python3
"""
Simple Inference Demo for Hair Triangle Detection
For paper review purposes only.

Usage:
    python demo_inference.py --image path/to/image.jpg --checkpoint path/to/model.pth
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import maximum_filter

from model_v3 import HairTriangleDetectorV3


def load_model(checkpoint_path, backbone='resnet50', use_cm=True, use_em=True, use_am=True, device='cuda'):
    """Load pretrained model"""
    model = HairTriangleDetectorV3(
        backbone_name=backbone,
        pretrained=False,
        use_cm=use_cm,
        use_em=use_em,
        use_am=use_am
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f" Model loaded from {checkpoint_path}")
    return model


def preprocess_image(image_path, image_size=(512, 512)):
    """Load and preprocess image"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # Resize
    image_resized = cv2.resize(image_rgb, image_size)
    
    # To tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image_rgb, (h, w)


def decode_predictions(heatmap, offset, angle, radius, width, threshold=0.3):
    """Decode model predictions to triangles"""
    # Convert to numpy
    heatmap_np = heatmap[0, 0].cpu().numpy()
    offset_np = offset[0].cpu().numpy()
    angle_np = angle[0].cpu().numpy()
    radius_np = radius[0, 0].cpu().numpy()
    width_np = width[0, 0].cpu().numpy()
    
    # Find peaks
    local_max = maximum_filter(heatmap_np, size=3)
    peaks = (heatmap_np == local_max) & (heatmap_np > threshold)
    
    y_coords, x_coords = np.where(peaks)
    
    triangles = []
    for x, y in zip(x_coords, y_coords):
        score = float(heatmap_np[y, x])
        
        # Get triangle parameters
        offset_xy = offset_np[:, y, x]
        apex_x = x + offset_xy[0]
        apex_y = y + offset_xy[1]
        
        angle_sin = angle_np[0, y, x]
        angle_cos = angle_np[1, y, x]
        angle_rad = np.arctan2(angle_sin, angle_cos)
        
        tri_radius = radius_np[y, x]
        tri_width = width_np[y, x]
        
        # Reconstruct triangle
        apex = np.array([apex_x, apex_y])
        base_center = apex + tri_radius * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        perp_angle = angle_rad + np.pi / 2
        base1 = base_center + (tri_width / 2) * np.array([np.cos(perp_angle), np.sin(perp_angle)])
        base2 = base_center - (tri_width / 2) * np.array([np.cos(perp_angle), np.sin(perp_angle)])
        
        triangle = np.array([apex, base1, base2])
        
        triangles.append({
            'triangle': triangle,
            'score': score
        })
    
    return triangles


def visualize_results(image, triangles, output_path, image_size=(512, 512), original_size=None):
    """Visualize detection results"""
    vis_image = image.copy()
    h_vis, w_vis = vis_image.shape[:2]
    
    # Scale factor
    if original_size:
        scale_x = w_vis / image_size[1]
        scale_y = h_vis / image_size[0]
    else:
        scale_x = scale_y = 1.0
    
    # Draw triangles
    for tri_info in triangles:
        triangle = tri_info['triangle'].copy()
        score = tri_info['score']
        
        # Scale to original size
        triangle[:, 0] *= scale_x
        triangle[:, 1] *= scale_y
        
        # Draw triangle edges
        pts = triangle.astype(np.int32)
        cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # Draw apex point
        apex = tuple(pts[0])
        cv2.circle(vis_image, apex, 4, (0, 255, 255), -1)
        
        # Draw score
        cv2.putText(vis_image, f'{score:.2f}', apex, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save result
    cv2.imwrite(str(output_path), vis_image)
    print(f" Result saved to {output_path}")
    
    return vis_image


def run_inference(model, image_path, checkpoint_path=None, output_dir='./demo_results', 
                 image_size=(512, 512), threshold=0.3, device='cuda'):
    """Run inference on single image"""
    # Load model
    if checkpoint_path:
        model = load_model(checkpoint_path, device=device)
    else:
        model = model.to(device)
        model.eval()
    
    # Preprocess
    image_tensor, original_image, original_size = preprocess_image(image_path, image_size)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Decode predictions
    triangles = decode_predictions(
        outputs['heatmap'],
        outputs['offset'],
        outputs['angle'],
        outputs['radius'],
        outputs['width'],
        threshold=threshold
    )
    
    print(f" Detected {len(triangles)} triangles")
    
    # Visualize
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / f"{Path(image_path).stem}_result.jpg"
    visualize_results(original_image, triangles, output_path, image_size, original_size)
    
    return {
        'triangles': triangles,
        'num_detections': len(triangles),
        'output_path': str(output_path)
    }


def main():
    parser = argparse.ArgumentParser(description='Hair Triangle Detection - Demo Inference')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./demo_results',
                       help='Output directory')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='Backbone architecture')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                       help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Detection threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--use_cm', action='store_true', default=True,
                       help='Use FCCM module')
    parser.add_argument('--use_em', action='store_true', default=True,
                       help='Use OTE module')
    parser.add_argument('--use_am', action='store_true', default=True,
                       help='Use TPA module')
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(
        args.checkpoint,
        backbone=args.backbone,
        use_cm=args.use_cm,
        use_em=args.use_em,
        use_am=args.use_am,
        device=device
    )
    
    # Run inference
    results = run_inference(
        model,
        args.image,
        image_size=tuple(args.image_size),
        threshold=args.threshold,
        output_dir=args.output,
        device=device
    )
    
    print("\n" + "="*60)
    print(f"Inference completed!")
    print(f"Detected: {results['num_detections']} triangles")
    print(f"Result saved: {results['output_path']}")
    print("="*60)


if __name__ == '__main__':
    main()

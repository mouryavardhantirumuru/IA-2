"""
AlexNet Feature Map & Filter Visualization
==========================================

This module provides a comprehensive pipeline to extract, analyze, and visualize:
1. First-layer convolutional filters (learned 11x11 RGB spatial kernels).
2. Intermediate feature maps across all 5 convolutional layers of AlexNet.
3. Hierarchical feature progression (from low-level edges to semantic abstractions).
4. Spatial activation heatmaps overlaid onto the original input image.
5. Quantitative metrics including layer sparsity (% zero activations), activation energy,
   and theoretical receptive fields.

Author: Tirumuru Mourya Vardhan
License: MIT
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import AlexNet_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class AlexNetFeatureVisualizer:
    """
    Visualizer and analysis engine for AlexNet feature maps and filters.
    
    Attributes:
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        model (torch.nn.Module): Pretrained AlexNet model in evaluation mode.
        layer_indices (Dict[str, int]): Mapping from layer names to indices in model.features.
        activations (Dict[str, torch.Tensor]): Cached forward activations from hooks.
        hooks (List[torch.utils.hooks.RemovableHandle]): List of registered forward hook handles.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initializes AlexNet model with pretrained ImageNet weights and registers forward hooks.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Using computation device: {self.device}")

        # Load AlexNet with DEFAULT weights (ImageNet-1K pretrained)
        print("[INFO] Loading pretrained AlexNet model...")
        weights = AlexNet_Weights.DEFAULT
        self.model = models.alexnet(weights=weights).to(self.device)
        self.model.eval()

        # ImageNet preprocessing transform matching AlexNet weights
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Raw display transform (unnormalized) for side-by-side visualization
        self.display_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])

        # Layer mapping in AlexNet.features
        # features[0]:  Conv1 (11x11, 64 channels, stride=4, pad=2)
        # features[3]:  Conv2 (5x5, 192 channels, stride=1, pad=2)
        # features[6]:  Conv3 (3x3, 384 channels, stride=1, pad=1)
        # features[8]:  Conv4 (3x3, 256 channels, stride=1, pad=1)
        # features[10]: Conv5 (3x3, 256 channels, stride=1, pad=1)
        self.target_layers: Dict[str, int] = {
            "Conv1": 0,
            "Conv2": 3,
            "Conv3": 6,
            "Conv4": 8,
            "Conv5": 10
        }

        # Theoretical receptive field specifications for AlexNet conv layers
        self.receptive_fields: Dict[str, Dict[str, float]] = {
            "Conv1": {"rf": 11, "stride": 4, "kernel": "11x11", "out_shape": (64, 55, 55)},
            "Conv2": {"rf": 51, "stride": 8, "kernel": "5x5",   "out_shape": (192, 27, 27)},
            "Conv3": {"rf": 99, "stride": 16, "kernel": "3x3",  "out_shape": (384, 13, 13)},
            "Conv4": {"rf": 131, "stride": 16, "kernel": "3x3", "out_shape": (256, 13, 13)},
            "Conv5": {"rf": 163, "stride": 16, "kernel": "3x3", "out_shape": (256, 13, 13)}
        }

        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attaches forward hooks to all target convolutional layers."""
        for layer_name, layer_idx in self.target_layers.items():
            layer = self.model.features[layer_idx]
            
            def get_hook(name):
                def hook(module, input_tensor, output_tensor):
                    self.activations[name] = output_tensor.detach()
                return hook

            handle = layer.register_forward_hook(get_hook(layer_name))
            self.hooks.append(handle)

    def remove_hooks(self) -> None:
        """Removes all registered hooks to avoid memory leaks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Loads and pre-processes an input image.

        Args:
            image_path (str): Filepath of the input image.

        Returns:
            Tuple[torch.Tensor, Image.Image]: Preprocessed tensor of shape (1, 3, 224, 224)
            and the cropped PIL Image for display.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        raw_image = Image.open(image_path).convert("RGB")
        tensor = self.transform(raw_image).unsqueeze(0).to(self.device)
        display_img = self.display_transform(raw_image)
        return tensor, display_img

    def forward_pass(self, input_tensor: torch.Tensor) -> Tuple[int, str, float]:
        """
        Executes a forward pass, caching intermediate feature maps and returning top-1 prediction.

        Args:
            input_tensor (torch.Tensor): Preprocessed input tensor.

        Returns:
            Tuple[int, str, float]: Predicted class index, label name, and probability percentage.
        """
        self.activations.clear()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, top_idx = torch.topk(probabilities, 1)

        class_id = top_idx[0].item()
        confidence = top_prob[0].item() * 100.0

        # Retrieve human-readable category name from weights meta
        weights = AlexNet_Weights.DEFAULT
        categories = weights.meta["categories"]
        label = categories[class_id] if class_id < len(categories) else f"Class_{class_id}"

        return class_id, label, confidence

    def visualize_conv1_filters(self, save_path: str) -> None:
        """
        Visualizes the 64 learned 11x11 RGB spatial filters of Conv1.
        These filters demonstrate learned Gabor-like edge, frequency, and color-opponent responses.

        Args:
            save_path (str): Output image destination path.
        """
        conv1_weights = self.model.features[0].weight.data.cpu().clone()
        # Shape: (64, 3, 11, 11)
        num_filters = conv1_weights.size(0)

        # Min-max normalize per filter to [0, 1] range for visualization
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))
        fig.suptitle(
            "AlexNet Layer 1 (Conv1) Learned Weight Filters (64 filters of size 11x11x3)\n"
            "Observation: Gabor-like directional edge detectors, frequency gratings, and color-opponency",
            fontsize=12, fontweight="bold", y=0.98
        )

        for i in range(num_filters):
            ax = axes[i // 8, i % 8]
            w = conv1_weights[i].permute(1, 2, 0).numpy()
            w_min, w_max = w.min(), w.max()
            if w_max > w_min:
                w_norm = (w - w_min) / (w_max - w_min)
            else:
                w_norm = w
            ax.imshow(w_norm)
            ax.axis("off")
            ax.set_title(f"F{i+1}", fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Conv1 weight filters visualization saved to: {save_path}")

    def visualize_layer_feature_maps(
        self,
        layer_name: str,
        save_path: str,
        num_channels: int = 64,
        cmap: str = "viridis"
    ) -> None:
        """
        Plots an 8x8 grid of feature maps for a specified convolutional layer.

        Args:
            layer_name (str): One of ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5'].
            save_path (str): Destination file path.
            num_channels (int): Number of channels to plot (default 64).
            cmap (str): Matplotlib colormap (default 'viridis').
        """
        if layer_name not in self.activations:
            raise KeyError(f"No activation found for {layer_name}. Run forward_pass() first.")

        act = self.activations[layer_name][0].cpu().numpy()
        total_channels, height, width = act.shape
        display_channels = min(num_channels, total_channels)

        # Rank channels by total activation energy (L1 sum) to show most active features
        energies = np.sum(np.abs(act), axis=(1, 2))
        top_indices = np.argsort(energies)[::-1][:display_channels]

        grid_rows = int(np.ceil(np.sqrt(display_channels)))
        grid_cols = int(np.ceil(display_channels / grid_rows))

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))
        axes = np.array(axes).reshape(grid_rows, grid_cols)

        rf_info = self.receptive_fields[layer_name]
        fig.suptitle(
            f"AlexNet {layer_name} Feature Maps (Top {display_channels} Active Channels)\n"
            f"Channel Resolution: {height}x{width} | Total Layer Channels: {total_channels} | "
            f"Receptive Field: {rf_info['rf']}x{rf_info['rf']} px",
            fontsize=12, fontweight="bold", y=0.98
        )

        for idx, ch in enumerate(top_indices):
            r = idx // grid_cols
            c = idx % grid_cols
            ax = axes[r, c]
            fmap = act[ch]
            im = ax.imshow(fmap, cmap=cmap)
            ax.axis("off")
            ax.set_title(f"Ch {ch}", fontsize=7)

        # Hide any unused subplots
        for idx in range(display_channels, grid_rows * grid_cols):
            r = idx // grid_cols
            c = idx % grid_cols
            axes[r, c].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {layer_name} feature maps saved to: {save_path}")

    def visualize_hierarchical_progression(
        self,
        display_img: Image.Image,
        save_path: str,
        predicted_label: str,
        confidence: float
    ) -> None:
        """
        Visualizes the progression of representations across all 5 convolutional layers
        side-by-side with the input image.

        Args:
            display_img (Image.Image): Cropped original input image.
            save_path (str): Destination file path.
            predicted_label (str): Predicted class label.
            confidence (float): Confidence score percentage.
        """
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()

        # Slot 0: Original Image
        axes[0].imshow(display_img)
        axes[0].set_title(
            f"Input Image (224x224)\nPred: {predicted_label} ({confidence:.1f}%)",
            fontsize=10, fontweight="bold"
        )
        axes[0].axis("off")

        # Slots 1-5: Conv1 to Conv5 mean activation energy maps
        layer_list = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]
        descriptions = [
            "Low-Level Edges & Gradients (55x55, RF=11px)",
            "Texture & Intersecting Patterns (27x27, RF=51px)",
            "Complex Structural Features (13x13, RF=99px)",
            "Object Parts & Spatial Context (13x13, RF=131px)",
            "High-Level Semantic Activations (13x13, RF=163px)"
        ]

        for i, layer_name in enumerate(layer_list):
            ax = axes[i + 1]
            act = self.activations[layer_name][0].cpu().numpy()
            # Mean activation energy across channels
            mean_map = np.mean(act, axis=0)
            # Normalize for visualization
            m_min, m_max = mean_map.min(), mean_map.max()
            if m_max > m_min:
                mean_norm = (mean_map - m_min) / (m_max - m_min)
            else:
                mean_norm = mean_map

            im = ax.imshow(mean_norm, cmap="inferno")
            ax.set_title(f"{layer_name} Energy Map\n{descriptions[i]}", fontsize=9, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            "AlexNet Hierarchical Feature Representation Progression\n"
            "Evolution from Low-Level Spatial Primitives to High-Level Semantic Invariance",
            fontsize=13, fontweight="bold", y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Hierarchical feature progression saved to: {save_path}")

    def visualize_heatmap_overlays(
        self,
        display_img: Image.Image,
        save_path: str
    ) -> None:
        """
        Projects each layer's spatial activation intensity directly onto the input image
        as a semi-transparent heatmap overlay to visualize spatial attention localization.

        Args:
            display_img (Image.Image): Cropped original input image.
            save_path (str): Destination file path.
        """
        img_np = np.array(display_img).astype(np.float32) / 255.0
        h_img, w_img, _ = img_np.shape

        fig, axes = plt.subplots(1, 6, figsize=(20, 4))

        axes[0].imshow(img_np)
        axes[0].set_title("Input Image", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        layer_names = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]
        for idx, name in enumerate(layer_names):
            ax = axes[idx + 1]
            act = self.activations[name][0].cpu().numpy()
            mean_act = np.mean(act, axis=0)

            # Resize heatmap to match image dimension
            heatmap_resized = Image.fromarray(mean_act).resize((w_img, h_img), resample=Image.BILINEAR)
            heat_arr = np.array(heatmap_resized)
            h_min, h_max = heat_arr.min(), heat_arr.max()
            if h_max > h_min:
                heat_norm = (heat_arr - h_min) / (h_max - h_min)
            else:
                heat_norm = heat_arr

            # Apply colormap
            colored_heat = cm.jet(heat_norm)[:, :, :3]
            # Blend original image and heatmap
            overlay = 0.5 * img_np + 0.5 * colored_heat

            ax.imshow(overlay)
            ax.set_title(f"{name} Overlay\n(RF: {self.receptive_fields[name]['rf']}px)", fontsize=9, fontweight="bold")
            ax.axis("off")

        fig.suptitle(
            "AlexNet Spatial Activation Heatmap Overlays\n"
            "Visualizing receptive field focus and spatial activation concentration across network depth",
            fontsize=12, fontweight="bold", y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Activation heatmap overlays saved to: {save_path}")

    def compute_layer_statistics(self, save_path: str) -> Dict[str, Dict[str, float]]:
        """
        Computes layer-wise metrics:
        - Activation Sparsity (% of neurons with zero activation due to ReLU)
        - Mean Activation Magnitude
        - Standard Deviation of Activations
        - Maximum Activation Response
        Generates and saves a statistical comparison chart.

        Args:
            save_path (str): Destination file path for bar chart.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of quantitative statistics per layer.
        """
        stats: Dict[str, Dict[str, float]] = {}
        layer_names = list(self.target_layers.keys())

        sparsities = []
        mean_energies = []
        max_responses = []

        for name in layer_names:
            act = self.activations[name][0].cpu().numpy()
            total_elements = act.size
            zero_elements = np.sum(act <= 1e-6)
            sparsity_pct = (zero_elements / total_elements) * 100.0
            mean_val = float(np.mean(np.abs(act)))
            std_val = float(np.std(act))
            max_val = float(np.max(act))

            stats[name] = {
                "sparsity_percent": sparsity_pct,
                "mean_activation": mean_val,
                "std_activation": std_val,
                "max_activation": max_val,
                "channels": act.shape[0],
                "spatial_shape": f"{act.shape[1]}x{act.shape[2]}"
            }

            sparsities.append(sparsity_pct)
            mean_energies.append(mean_val)
            max_responses.append(max_val)

        # Plot quantitative comparison charts
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

        # 1. Sparsity
        bars1 = ax1.bar(layer_names, sparsities, color="#3498db", edgecolor="#2980b9")
        ax1.set_title("Layer-wise Activation Sparsity (%)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Sparsity (% zeros)", fontsize=10)
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

        # 2. Mean Activation Energy
        bars2 = ax2.bar(layer_names, mean_energies, color="#e67e22", edgecolor="#d35400")
        ax2.set_title("Mean Absolute Activation Energy", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Activation", fontsize=10)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

        # 3. Maximum Response
        bars3 = ax3.bar(layer_names, max_responses, color="#2ecc71", edgecolor="#27ae60")
        ax3.set_title("Peak Activation Response", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Max Activation", fontsize=10)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.1f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(
            "Quantitative Analysis of AlexNet Convolutional Activations\n"
            "Evaluating Activation Sparsity, Average Energy, and Peak Firing Rates",
            fontsize=12, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Activation statistics chart saved to: {save_path}")

        return stats

    def run_full_pipeline(self, image_path: str, results_dir: str) -> Dict[str, any]:
        """
        Executes the end-to-end visualization and analysis pipeline.

        Args:
            image_path (str): Path to input image.
            results_dir (str): Directory where output figures will be written.

        Returns:
            Dict[str, any]: Summary metrics and outputs.
        """
        print(f"\n{'='*70}")
        print(f"RUNNING ALEXNET FEATURE MAP VISUALIZATION PIPELINE")
        print(f"Target Image: {image_path}")
        print(f"Output Directory: {results_dir}")
        print(f"{'='*70}\n")

        # 1. Preprocess image
        input_tensor, display_img = self.preprocess_image(image_path)

        # 2. Forward pass and classification
        class_id, label, confidence = self.forward_pass(input_tensor)
        print(f"[PREDICTION] Top Class: {label} (Class ID: {class_id}) | Confidence: {confidence:.2f}%")

        # 3. Visualize Conv1 learned filters
        self.visualize_conv1_filters(os.path.join(results_dir, "conv1_filters_visualization.png"))

        # 4. Visualize feature maps for all 5 conv layers
        for layer_name in ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]:
            save_name = f"{layer_name.lower()}_feature_maps.png"
            self.visualize_layer_feature_maps(
                layer_name,
                os.path.join(results_dir, save_name),
                num_channels=64,
                cmap="viridis"
            )

        # 5. Visualize cross-layer hierarchical progression
        self.visualize_hierarchical_progression(
            display_img,
            os.path.join(results_dir, "layer_progression_comparison.png"),
            label,
            confidence
        )

        # 6. Visualize heatmap overlays
        self.visualize_heatmap_overlays(
            display_img,
            os.path.join(results_dir, "activation_heatmaps_overlay.png")
        )

        # 7. Compute and plot activation statistics
        stats = self.compute_layer_statistics(
            os.path.join(results_dir, "activation_statistics.png")
        )

        print("\n[SUMMARY STATISTICS]")
        print(f"{'Layer':<8} | {'Shape':<12} | {'Sparsity (%)':<14} | {'Mean Energy':<12} | {'Max Peak':<10}")
        print("-" * 65)
        for l_name, s in stats.items():
            print(f"{l_name:<8} | {s['spatial_shape']:<12} | {s['sparsity_percent']:<14.2f} | {s['mean_activation']:<12.3f} | {s['max_activation']:<10.2f}")

        print(f"\n[SUCCESS] Pipeline completed successfully! All visual artifacts saved to {results_dir}\n")
        return {
            "prediction": {"class_id": class_id, "label": label, "confidence": confidence},
            "statistics": stats
        }


def main():
    parser = argparse.ArgumentParser(description="AlexNet Feature Map and Filter Visualizer")
    parser.add_argument(
        "--image",
        type=str,
        default="dataset/dog.jpg",
        help="Path to input image file (default: dataset/dog.jpg)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save generated visualizations (default: results)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cpu' or 'cuda')"
    )

    args = parser.parse_args()

    visualizer = AlexNetFeatureVisualizer(device=args.device)
    visualizer.run_full_pipeline(image_path=args.image, results_dir=args.output_dir)
    visualizer.remove_hooks()


if __name__ == "__main__":
    main()

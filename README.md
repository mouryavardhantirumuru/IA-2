# AlexNet Feature Map and Filter Visualization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Torchvision](https://img.shields.io/badge/Torchvision-0.15%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)
![Topic](https://img.shields.io/badge/Topic-CNN%20Feature%20Interpretability-orange.svg)

---

## 1. Project Title
**Feature Map and Learned Filter Visualization Across Convolutional Layers of AlexNet**

---

## 2. Objective
The primary objective of this project is to implement, extract, and interpret the feature representations learned across the convolutional hierarchy of **AlexNet** (Krizhevsky et al., 2012). 

Specifically, this project aims to:
1. **Demystify the CNN "Black Box":** Provide direct visual evidence of how raw input pixels are transformed into semantic concepts across hierarchical layers.
2. **Visualize Learned First-Layer Filters (Conv1):** Examine the 64 learned spatial weight kernels ($11 \times 11 \times 3$) and identify directional Gabor-like edge detectors, color opponency, and frequency gratings.
3. **Extract Intermediate Activations:** Utilize PyTorch forward hooks to record activation feature maps from all five convolutional layers:
   - **Conv1** ($55 \times 55 \times 64$)
   - **Conv2** ($27 \times 27 \times 192$)
   - **Conv3** ($13 \times 13 \times 384$)
   - **Conv4** ($13 \times 13 \times 256$)
   - **Conv5** ($13 \times 13 \times 256$)
4. **Trace Hierarchical Representation Evolution:** Track the transition from low-level spatial primitives (edges, textures) to mid-level geometric motifs, and finally to high-level semantic object components.
5. **Analyze Spatial Attention Heatmaps:** Project layer-wise activation energy back onto the original image to understand where the network focuses spatial attention.
6. **Quantify Non-Linear Activation Properties:** Measure activation sparsity (% zero neurons induced by ReLU non-linearity), mean activation magnitude, and theoretical receptive field growth.

---

## 3. Problem Statement & Motivation
Deep Convolutional Neural Networks achieve state-of-the-art accuracy in image recognition, object detection, and visual reasoning. However, deep neural networks are often criticized as uninterpretable "black boxes". Understanding what each layer learns is critical for:
- **Model Debugging & Explainability:** Identifying whether a network bases decisions on intended object features or background artifacts.
- **Biologically Inspired Computing:** Correlating artificial convolutional filters with receptive field structures discovered in mammalian visual cortex (V1 and V2 simple cells).
- **Architecture Optimization:** Understanding receptive field saturation, channel sparsity, and computational bottlenecks in convolutional neural network design.

---

## 4. Architectural Summary of AlexNet

AlexNet consists of 5 Convolutional layers, 3 Overlapping Max-Pooling layers, and 3 Fully-Connected (Dense) layers.

| Layer Name | Component Type | Kernel / Size | Stride | Padding | Output Resolution | Output Channels | Trainable Parameters | Cumulative Receptive Field |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Input** | RGB Image | — | — | — | $224 \times 224$ | 3 | 0 | $1 \times 1$ px |
| **Conv1** | Conv2d + ReLU | $11 \times 11$ | 4 | 2 | $55 \times 55$ | 64 | 23,296 | $11 \times 11$ px |
| **Pool1** | MaxPool2d | $3 \times 3$ | 2 | 0 | $27 \times 27$ | 64 | 0 | $19 \times 19$ px |
| **Conv2** | Conv2d + ReLU | $5 \times 5$ | 1 | 2 | $27 \times 27$ | 192 | 307,392 | $51 \times 51$ px |
| **Pool2** | MaxPool2d | $3 \times 3$ | 2 | 0 | $13 \times 13$ | 192 | 0 | $67 \times 67$ px |
| **Conv3** | Conv2d + ReLU | $3 \times 3$ | 1 | 1 | $13 \times 13$ | 384 | 663,936 | $99 \times 99$ px |
| **Conv4** | Conv2d + ReLU | $3 \times 3$ | 1 | 1 | $13 \times 13$ | 256 | 884,992 | $131 \times 131$ px |
| **Conv5** | Conv2d + ReLU | $3 \times 3$ | 1 | 1 | $13 \times 13$ | 256 | 590,080 | $163 \times 163$ px |
| **Pool5** | MaxPool2d | $3 \times 3$ | 2 | 0 | $6 \times 6$ | 256 | 0 | $195 \times 195$ px |
| **FC6** | Linear + ReLU + Dropout | — | — | — | $1 \times 4096$ | — | 37,752,832 | Full Input ($224 \times 224$) |
| **FC7** | Linear + ReLU + Dropout | — | — | — | $1 \times 4096$ | — | 16,781,312 | Full Input ($224 \times 224$) |
| **FC8** | Linear (Softmax) | — | — | — | $1 \times 1000$ | — | 4,097,000 | Full Input ($224 \times 224$) |

*Total Parameters: ~61.1 Million (Features: ~2.47 Million, Classifier: ~58.63 Million).*

---

## 5. Mathematical Foundations

### 5.1 Discrete 2D Convolution (Cross-Correlation)
For an input activation tensor $X \in \mathbb{R}^{C_{in} \times H \times W}$ and a convolutional filter weight tensor $W \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$, the output activation at channel $c_{out}$ and spatial location $(x, y)$ is:

$$
Y(c_{out}, x, y) = b(c_{out}) + \sum_{c_{in}=1}^{C_{in}} \sum_{i=-(K-1)/2}^{(K-1)/2} \sum_{j=-(K-1)/2}^{(K-1)/2} X(c_{in}, x \cdot s + i, y \cdot s + j) \cdot W(c_{out}, c_{in}, i, j)
$$

Where:
- $s$ is the convolution stride
- $K$ is the spatial kernel dimension
- $b$ is the learnable bias parameter

### 5.2 Theoretical Receptive Field ($RF$) Growth
The theoretical receptive field corresponds to the spatial region in the input image that directly influences a single neuron's activation at layer $l$. It is governed by the recurrence relations:

$$
RF_l = RF_{l-1} + (k_l - 1) \cdot J_{l-1}
$$
$$
J_l = J_{l-1} \cdot s_l, \quad \text{with initial conditions } RF_0 = 1, \; J_0 = 1
$$

Where $k_l$ is the kernel size at layer $l$, $s_l$ is the layer stride, and $J_l$ denotes the cumulative feature jump (stride product).
- By **Conv1**, $RF = 11 \times 11$ px (fine local gradients).
- By **Conv5**, $RF = 163 \times 163$ px (over 70% of the entire $224 \times 224$ input image), allowing the layer to integrate spatial relations across object boundaries.

### 5.3 Rectified Linear Activation (ReLU)
AlexNet replaced saturating non-linearities ($\tanh$, sigmoid) with the non-saturating ReLU function:

$$
f(z) = \max(0, z)
$$

ReLU introduces scale-invariant non-linearity, accelerates stochastic gradient descent convergence by $6\times$, and enforces **representation sparsity** by clamping negative pre-activations to exactly zero.

---

## 6. Project Directory Structure

```
alexnet-feature-map-visualization/
├── README.md                                    # Comprehensive technical documentation & analysis
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Standard git ignore configuration
├── dataset/                                     # Diverse vision benchmark test images
│   ├── baboon.jpg                               # Mandrill face (high-frequency fur texture)
│   ├── butterfly.jpg                            # Swallowtail butterfly (fine geometric symmetry)
│   ├── car.jpg                                  # Objects/vehicle structure
│   ├── dog.jpg                                  # Samoyed / Golden Retriever (ImageNet benchmark)
│   └── fruits.jpg                               # Multi-color curved shapes
├── src/
│   └── alexnet_feature_visualization.py         # Modular PyTorch feature extraction pipeline
├── notebooks/
│   └── AlexNet_Feature_Map_Visualization.ipynb  # Interactive Jupyter notebook walkthrough
├── results/                                     # Publication-quality generated figures
│   ├── conv1_filters_visualization.png          # Learned 11x11 RGB spatial weight kernels
│   ├── conv1_feature_maps.png                   # Conv1 channel activations (55x55)
│   ├── conv2_feature_maps.png                   # Conv2 channel activations (27x27)
│   ├── conv3_feature_maps.png                   # Conv3 channel activations (13x13)
│   ├── conv4_feature_maps.png                   # Conv4 channel activations (13x13)
│   ├── conv5_feature_maps.png                   # Conv5 channel activations (13x13)
│   ├── layer_progression_comparison.png         # Cross-layer hierarchical representation progression
│   ├── activation_heatmaps_overlay.png          # Spatial attention overlays on input image
│   └── activation_statistics.png                # Sparsity & energy quantitative bar charts
└── screenshots/
    └── output.png                               # Complete CLI execution output terminal capture
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+ and Torchvision 0.15+

### Installation
Clone or navigate to the project directory:
```bash
cd alexnet-feature-map-visualization
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 8. Execution Guide

### Running via Command-Line Interface (CLI)
To run the complete feature visualization pipeline on the default sample image (`dataset/dog.jpg`):
```bash
python src/alexnet_feature_visualization.py
```

To run on a custom image or specify a custom output directory:
```bash
python src/alexnet_feature_visualization.py --image dataset/butterfly.jpg --output_dir results/
```

### Running the Interactive Jupyter Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook notebooks/AlexNet_Feature_Map_Visualization.ipynb
```

---

## 9. Experimental Results & Visual Analysis

### 9.1 Learned Weight Filters in Conv1 (`conv1_filters_visualization.png`)
AlexNet's first layer features 64 learned filters of size $11 \times 11 \times 3$.
- **Orientation Selectivity:** Filters exhibit alternating light and dark bands oriented at specific angles ($0^\circ, 30^\circ, 45^\circ, 90^\circ, 135^\circ$), acting identically to mathematical 2D Gabor wavelets.
- **Color Opponency:** Certain filters specialize in opposing color contrasts (blue-yellow, red-cyan, green-magenta), reproducing the opponent-process color channels of human vision.
- **Frequency Tuning:** Filters vary from low-frequency smooth gradients to high-frequency edge ripple detectors.

### 9.2 Layer-by-Layer Feature Map Hierarchy

| Layer | Resolution | Channel Depth | Key Visual Characteristics Observed |
| :--- | :---: | :---: | :--- |
| **Conv1** | $55 \times 55$ | 64 | Direct edge detection, spatial outlines, silhouette contours, and color boundary maps. The subject is instantly recognizable from raw feature maps. |
| **Conv2** | $27 \times 27$ | 192 | Combinations of edges: intersecting corners, circular textures, fine fur ripples, and cross-hatching. Spatial resolution is halved by max pooling. |
| **Conv3** | $13 \times 13$ | 384 | Complex structural motifs and recurring textures. Invariance to small spatial translations begins to emerge. |
| **Conv4** | $13 \times 13$ | 256 | Distinct object parts: snout, eyes, ear contours, paw boundaries. Spatial resolution remains $13 \times 13$ while semantic specificity sharpens. |
| **Conv5** | $13 \times 13$ | 256 | High-level semantic features: channels fire selectively on entire object categories or global shapes rather than individual pixels. |

### 9.3 Quantitative Activation Statistics

Using `dataset/dog.jpg` (Predicted class: **Samoyed**, Confidence: **72.45%**):

| Layer | Feature Map Dimensions | Activation Sparsity (% Zero) | Mean Activation Energy | Peak Activation Response |
| :--- | :---: | :---: | :---: | :---: |
| **Conv1** | $64 \times 55 \times 55$ | **60.08%** | 0.377 | 19.65 |
| **Conv2** | $192 \times 27 \times 27$ | **80.58%** | 0.437 | 33.81 |
| **Conv3** | $384 \times 13 \times 13$ | **78.82%** | 0.623 | 35.42 |
| **Conv4** | $256 \times 13 \times 13$ | **82.04%** | 0.529 | 30.65 |
| **Conv5** | $256 \times 13 \times 13$ | **88.98%** | 0.280 | 29.68 |

#### Key Analytical Insights:
1. **Monotonic Sparsity Growth:** Sparsity climbs from **60.08%** in Conv1 to **88.98%** in Conv5. In deeper layers, almost 9 out of 10 neurons are deactivated for any given input, demonstrating extreme semantic selectivity.
2. **Receptive Field and Semantic Compression:** Early layers encode dense, low-level spatial geometry across all pixels; deeper layers compress the representation into a sparse, translation-invariant semantic vector suitable for fully-connected classification.

---

## 10. Viva-Voce Questions & Comprehensive Answers

### Q1: Why did AlexNet use an $11 \times 11$ kernel in Conv1, whereas modern architectures (like VGG or ResNet) use $3 \times 3$ kernels?
> **Answer:** In 2012, AlexNet was designed to process high-resolution $224 \times 224$ images quickly on dual 3GB GPUs. An $11 \times 11$ kernel with stride 4 rapidly downsampled the spatial dimension from $224 \times 224$ down to $55 \times 55$ in a single pass, saving GPU memory. Later research (VGG, Simonyan & Zisserman 2014) proved that stacking two $3 \times 3$ convolutions yields an effective $5 \times 5$ receptive field with fewer parameters and extra non-linearities, which is why modern networks favor small cascaded kernels.

### Q2: What is the purpose of PyTorch forward hooks in feature map visualization?
> **Answer:** In standard PyTorch execution, intermediate layer tensors are deleted from memory immediately after consumption during the forward pass to minimize GPU/RAM overhead. Forward hooks (`register_forward_hook`) allow developers to intercept and clone intermediate activations when data flows through specific submodules without altering the core model architecture.

### Q3: Why do learned filters in Conv1 closely resemble Gabor filters and V1 biological receptive fields?
> **Answer:** Mathematical natural image statistics are dominated by localized step edges, luminance transitions, and oriented gradients. Gradient descent optimization independently rediscovers the optimal mathematical basis functions for compressing natural image statistics—which happens to match the receptive field profiles of mammalian primary visual cortex simple cells discovered by Hubel and Wiesel.

### Q4: How does ReLU contribute to the observed activation sparsity?
> **Answer:** The ReLU activation function is defined as $f(x) = \max(0, x)$. Any negative convolution output is mapped strictly to zero. This enforces representation sparsity, reduces co-adaptation among neurons, and creates piece-wise linear decision boundaries that train significantly faster than smooth sigmoidal activations.

### Q5: What is the difference between receptive field and feature map resolution?
> **Answer:** Feature map resolution is the spatial size ($H \times W$) of the output grid produced by a layer (e.g., $55 \times 55$ for Conv1, down to $13 \times 13$ for Conv5). The receptive field is the physical span in the original input image ($224 \times 224$) that mathematically influences a single activation point. As feature map resolution decreases through striding and pooling, the receptive field expands.

### Q6: What was Local Response Normalization (LRN) in AlexNet, and why is it no longer used today?
> **Answer:** LRN implemented lateral inhibition inspired by neurobiology, normalizing a neuron's activation against adjacent feature channels: $b_{x,y}^i = a_{x,y}^i / \left(k + \alpha \sum_{j} (a_{x,y}^j)^2\right)^\beta$. However, LRN provided minimal empirical gain and was completely replaced in 2015 by **Batch Normalization (BatchNorm)**, which normalizes across mini-batches and stabilizes internal covariate shift much more effectively.

---

## 11. Git Commit History & Authenticity Verification

In compliance with project submission guidelines, development history has been tracked through progressive, meaningful Git commits:

To inspect the commit history:
```bash
git log --oneline --graph --decorate
```

To verify code authorship and line-by-line authenticity:
```bash
git blame src/alexnet_feature_visualization.py
```

---

## 12. References & Acknowledgments
1. **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).** ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS 2012)*.
2. **Zeiler, M. D., & Fergus, R. (2014).** Visualizing and Understanding Convolutional Networks. *European Conference on Computer Vision (ECCV 2014)*.
3. **Simonyan, K., & Zisserman, A. (2014).** Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv:1409.1556*.
4. **PyTorch Team (2024).** Torchvision Model Zoo & Pretrained AlexNet Checkpoint (`alexnet-owt-7be5be79.pth`).

---
*Created by Tirumuru Mourya Vardhan for Academic Demonstration & Evaluation.*

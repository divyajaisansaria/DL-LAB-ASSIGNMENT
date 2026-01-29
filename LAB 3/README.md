# Comparative Study of CNN Architectures, Loss Functions, and Optimization Strategies

## Project Overview
This project focuses on implementing, training, and evaluating multiple landmark Convolutional Neural Network (CNN) architectures to analyze the impact of **network depth, dataset complexity, loss functions, and optimization strategies** on classification performance and convergence behavior.

The work is divided into two parts:
- **Part 1:** CNN architecture comparison on a complex dataset (CIFAR-10)
- **Part 2:** Loss function and optimizer impact analysis on a simple dataset (MNIST)

All experiments are implemented using **PyTorch** and executed on **Google Colab with GPU support**.

---

## Datasets Used

### Part 1 Dataset
- **CIFAR-10**
  - 32×32 RGB images
  - 10 object classes
  - Complex and noisy dataset used to analyze the effect of network depth and architecture

### Part 2 Dataset
- **MNIST**
  - 28×28 grayscale handwritten digit images
  - 10 balanced classes
  - Simple dataset used to study convergence behavior of different loss functions

---

## Part 1: CNN Architecture Comparison (CIFAR-10)

### Objective
To analyze how **network depth and architectural complexity** affect:
- Classification accuracy
- Convergence behavior
- Computational efficiency

### Implemented Architectures
- LeNet-5  
- AlexNet  
- VGGNet  
- ResNet-50  
- ResNet-100  
- EfficientNet  
- InceptionV3  
- MobileNet  

### Experimental Setup
- CIFAR-10 dataset with normalization and data augmentation
- Same evaluation protocol across all models
- Multiple configurations of learning rate, optimizer, and epochs
- Training performed under identical conditions for fair comparison

### Evaluation Metrics
- Training Accuracy
- Testing Accuracy
- Convergence Speed
- Architectural Depth vs Performance

---

## Part 2: Loss Function and Optimizer Impact Study (MNIST)

### Objective
To study how **different loss functions and optimization strategies** influence:
- Convergence speed
- Final training and testing accuracy

### Experimental Configuration

| Model     | Optimizer | Epochs | Loss Function |
|----------|----------|--------|---------------|
| VGGNet   | Adam     | 10     | Binary Cross-Entropy (BCE) |
| AlexNet  | SGD      | 20     | Focal Loss |
| ResNet   | Adam     | 15     | ArcFace Loss |

### Loss Functions Used
- **Binary Cross-Entropy (BCE):** One-vs-all formulation for multi-class classification
- **Focal Loss:** Focuses learning on hard samples
- **ArcFace Loss:** Margin-based loss that enforces class separability in embedding space

---

## Results and Visualizations

### Part 1 Results: CNN Architecture Comparison on CIFAR-10
The following figure compares the classification accuracy of different CNN architectures on the CIFAR-10 dataset, highlighting the impact of network depth and architectural design.

![CIFAR-10 CNN Accuracy Comparison](results/accuracy_plots.pngaccuracy_plots.png)

---

### Part 2 Results: Loss Function Convergence on MNIST
The following figure illustrates the convergence behavior of different loss functions and optimization strategies on the MNIST dataset.

![MNIST Convergence Comparison](results/convergence_curves.pngconvergence_curves.png)

---

## How to Run
1. Open the notebook files in Google Colab
2. Enable GPU runtime
3. Run all cells sequentially
4. Modify dataset or configurations if required

---

## Conclusion
This study demonstrates that **dataset complexity strongly influences model design choices**. While deep architectures and residual connections are essential for complex datasets like CIFAR-10, simpler datasets such as MNIST converge efficiently even with basic loss functions. Advanced loss functions and optimizers become increasingly valuable as task complexity increases.

---

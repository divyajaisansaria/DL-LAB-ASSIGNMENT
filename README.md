# Deep Learning Lab Assignments

This repository contains three comprehensive lab assignments focused on Deep Learning concepts, particularly Convolutional Neural Networks (CNNs) and their applications in image classification tasks.

## Repository Structure

```
DL-LAB-ASSIGNMENT/
├── LAB 1/          # Neural Network Fundamentals
├── LAB 2/          # CNN Training Optimization
└── LAB 3/          # Advanced CNN Architectures
```

---

## LAB 1: Introduction to Neural Networks

### Overview
Foundational lab focusing on basic neural network concepts and implementations using Deep Learning frameworks.

### Key Topics
- Neural network fundamentals
- Basic model training and evaluation
- Introduction to Jupyter notebooks for ML experiments

### Files
- `DL_LABB1.ipynb` - Main notebook with experiments and implementations

---

## LAB 2: CNN Training and Optimization (MNIST)

### Overview
Comprehensive analysis of **activation functions**, **optimizers**, and **regularization techniques** on neural network performance using the MNIST handwritten digit classification dataset.

### Key Experiments

#### 1. Activation Function Comparison
- **Functions Tested:** Sigmoid, Tanh, ReLU
- **Best Performer:** ReLU (97.35% validation accuracy)
- **Key Finding:** ReLU achieves fastest convergence and best accuracy

#### 2. Optimizer Showdown
- **Optimizers Tested:** SGD, SGD with Momentum, Adam
- **Best Performer:** Adam (99.24% validation accuracy)
- **Key Finding:** Adam optimizer provides fastest convergence

#### 3. Regularization Techniques
- **Techniques Tested:** Dropout, Batch Normalization
- **Best Configuration:** Batch Normalization + Dropout (99.10% accuracy)
- **Key Finding:** Combined regularization improves generalization

### Dataset
- **MNIST**: 60,000 training images, 10,000 test images (28×28 grayscale)

### Files
- `DL_LAB__2.ipynb` - Main notebook with all experiments
- `README.md` - Detailed results and analysis
- `results/` - Visualization plots

---

## LAB 3: Advanced CNN Architectures and Loss Functions

### Overview
Comparative study of state-of-the-art **CNN architectures**, **loss functions**, and **optimization strategies** to understand their impact on classification performance and feature representation quality.

### Key Components

#### Part 1: CNN Architecture Comparison (CIFAR-10)
- **Architectures Implemented:**
  - LeNet-5
  - AlexNet
  - VGGNet
  - ResNet-50 / ResNet-100
  - EfficientNet
  - InceptionV3
  - MobileNet

- **Focus:** Network depth vs performance trade-offs

#### Part 2: Loss Function and Optimizer Analysis (MNIST)
- **Loss Functions:** Binary Cross-Entropy (BCE), Focal Loss, ArcFace Loss
- **Optimizers:** Adam, SGD
- **Focus:** Convergence behavior and accuracy

#### Part 3: Feature Space Visualization (CIFAR-10)
- **Method:** t-SNE visualization of learned feature embeddings
- **Comparison:** Softmax vs ArcFace loss
- **Key Finding:** ArcFace produces more compact and well-separated feature clusters

### Datasets
- **CIFAR-10**: 32×32 RGB images across 10 classes
- **MNIST**: 28×28 grayscale handwritten digits

### Files
- `Lab3_CNN_Comparison (1).ipynb` - Architecture comparison experiments
- `LAB3_PART2.ipynb` - Loss function and optimizer analysis
- `LAB3_PART3.ipynb` - Feature space visualization
- `README.md` - Detailed results and analysis
- `results/` - Visualization plots and charts

---

## Technologies Used

- **Framework:** PyTorch
- **Platform:** Google Colab with GPU support
- **Tools:** Jupyter Notebooks
- **Visualization:** Matplotlib, t-SNE

---

## How to Run

1. Clone this repository
2. Open the desired lab folder
3. Upload the `.ipynb` file to Google Colab
4. Enable GPU runtime (Runtime → Change runtime type → GPU)
5. Run all cells sequentially

---

## Key Learnings

### From LAB 2:
- ReLU activation function is most effective for CNNs
- Adam optimizer provides fastest convergence
- Combining Batch Normalization with Dropout improves generalization

### From LAB 3:
- Deeper architectures generally improve performance on complex datasets
- Loss function choice critically impacts feature quality
- Margin-based losses (e.g., ArcFace) enhance feature separability
- Different architectures have different computational trade-offs

---

## Author

Divya Jaisansaria

---

## License

This repository is for educational purposes as part of Deep Learning coursework.

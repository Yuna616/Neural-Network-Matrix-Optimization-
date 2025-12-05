# Neural Network Matrix Optimization
*A C++ project focused on implementing and optimizing matrix-based neural network operations.*

This repository explores how neural network layers—especially Fully Connected (FC) layers—can be implemented and optimized at a low level using C++.  
Rather than relying on high-level frameworks such as PyTorch or TensorFlow, this project aims to deepen understanding by manually working with core matrix operations.

---

## 🚀 Features

### ✔ Fully Connected Layer Implementation (`fc_layer.cpp`)
- Direct computation of:
  - `output = weight * input + bias`
- Modular design allowing experimentation with:
  - Loop unrolling  
  - Memory access optimization  
  - SIMD / parallelization  
  - Matrix layout changes

### ✔ Executable Entry Point (`main.cpp`)
- Loads weights/bias from binary files  
- Performs forward computation using the FC layer  
- Provides baseline structure for expanding into multi-layer models

### ✔ Pretrained Weight Included
- `vgg19.w24.bias.bin`  
  - Contains sample FC-layer bias values from the VGG19 architecture  
  - Useful for testing real-world inference scenarios

---

## 📂 Project Structure

```

Neural-Network-Matrix-Optimization-/
│
├── main.cpp                # Application entry point
├── fc_layer.cpp            # Fully Connected Layer implementation
├── Makefile                # Build instructions
├── data/                   # Model weights and data files
├── vgg19.w24.bias.bin      # Sample pretrained weights
└── .gitignore

````

---

## 🛠 Build Instructions

Build using the included Makefile:

```bash
make
````

Run the compiled binary:

```bash
./run
```

---

## 🧪 Usage

The current version focuses on testing the forward pass of a Fully Connected layer.

Typical workflow:

1. Load weight and bias files
2. Provide an input vector
3. Run forward computation
4. Output results

Planned expansions include:

* Multiple FC layers
* Activation functions
* Benchmarking matrix optimization techniques
* Inference demos for datasets like MNIST or CIFAR




도 만들어줄게!
```

# Exploring Convolutional Layers Through Data and Experiments

## Problem Description

This project explores convolutional layers as an architectural choice for image
classification, rather than treating neural networks as black boxes. The goal is
to understand how architectural decisions — specifically convolution versus fully
connected layers, and the depth of a convolutional stack — affect a model's
ability to learn from image data.

To do this, three things were built and compared:
1. A baseline neural network using only Dense (fully connected) layers.
2. A convolutional neural network (CNN) designed from scratch, with each
   architectural choice explicitly justified.
3. A controlled experiment isolating convolutional depth (1 vs. 2 vs. 3
   convolutional layers), with every other setting held fixed.

The final CNN was also trained on AWS SageMaker to demonstrate the training
workflow in a cloud environment.

## Dataset Description

**Dataset:** CIFAR-10 (via `tf.keras.datasets.cifar10`)

- 60,000 color images, 32×32 pixels, 3 channels (RGB)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images / 10,000 test images
- Perfectly balanced: 5,000 images per class in training, 1,000 per class in test

**Why CIFAR-10 is appropriate for convolutional layers:**
- **Color matters.** Unlike grayscale datasets (e.g. MNIST), CIFAR-10's 3 color
  channels let convolutional filters learn color-based patterns as well as shape.
- **Objects vary in pose, position, and background**, unlike centered, uniform
  digit images — this variability is exactly what convolution's spatial
  pattern-matching is suited to handle.
- **Real objects have hierarchical structure** (edges -> parts -> whole objects),
  which benefits from stacking multiple convolutional layers.

**Preprocessing applied:**
- Pixel values rescaled from integers (0–255) to floats (0–1) for stable training.
- Images flattened to a 3,072-length vector *only* for the baseline Dense model.
- Images kept in their native (32, 32, 3) shape for all convolutional models,
  since convolution relies on knowing which pixels are spatial neighbors.
- 5,000 training images set aside as a validation set; the 10,000-image test set
  was used only for final evaluation.

## Architecture Diagrams

**Baseline (Flatten + Dense):**

Input (32x32x3)
-> Flatten -> (3072,)
-> Dense(256, relu)
-> Dense(128, relu)
-> Dense(10, linear) [820,874 params]


**Final CNN (3 convolutional blocks):**

Input (32x32x3)
-> Conv2D(32, 3x3, same, relu) -> MaxPool(2x2) [32x32 -> 16x16]
-> Conv2D(64, 3x3, same, relu) -> MaxPool(2x2) [16x16 -> 8x8]
-> Conv2D(128, 3x3, same, relu) -> MaxPool(2x2) [8x8 -> 4x4]
-> Flatten -> (2048,)
-> Dense(128, relu)
-> Dense(10, linear) [356,810 params]


**Design choices and justification:**
- **Kernel size 3x3** — standard choice; two stacked 3x3 layers approximate a 5x5
  receptive field with fewer parameters and an added non-linearity.
- **Padding "same"** — preserves spatial size after each convolution, avoiding
  early loss of border information on an already-small 32x32 image.
- **Stride 1** — checks every position for thorough feature extraction; spatial
  shrinking is handled separately by pooling.
- **Filters double each block (32 -> 64 -> 128)** — deeper layers detect a larger
  number of more complex, combined patterns, so they benefit from more filters.
- **ReLU activation** — standard for hidden layers, trains faster than sigmoid.
- **2x2 max pooling per block** — shrinks the feature map (less downstream
  computation) and adds tolerance to small shifts in feature position.

## Experimental Results

### Baseline vs. Final CNN

| Metric | Baseline (Dense) | CNN (3 conv layers) |
|---|---|---|
| Test accuracy | 47.9% | 73.1% |
| Total parameters | 820,874 | 356,810 |
| Validation behavior | Plateaus ~epoch 6-7 | Overfits after ~epoch 8-9 |

### Controlled experiment: convolutional depth

All models share identical settings (3x3 kernels, "same" padding, stride 1, ReLU,
2x2 max pooling per block, same Dense head, Adam optimizer at lr=1e-3, sparse
categorical crossentropy, 15 epochs, batch size 64, identical train/validation
split). Only the number of convolutional blocks varies.

| Depth | Test accuracy | Test loss | Total params | Approx. time/epoch |
|---|---|---|---|---|
| 1 conv layer | 64.6% | 1.075 | — | ~15-16s |
| 2 conv layers | 69.2% | 1.353 | 545,098 | ~24-26s |
| 3 conv layers | 73.1% | 1.226 | 356,810 | ~33-38s |

**Key observations:**
- Test accuracy increased monotonically with depth, but test loss did not — the
  2-layer model had the worst loss despite better accuracy than the 1-layer
  model, indicating it became more confidently wrong on some mistakes even while
  classifying more examples correctly overall.
- Overfitting severity was not simply proportional to depth: the 1-layer model
  overfit least, the 2-layer model overfit most severely, and the 3-layer model
  overfit less than the 2-layer model — likely because its final feature map
  (4x4) is small enough to limit how much fine, memorizable detail remains.
- Parameter count did not increase monotonically with depth. Each added conv
  layer is cheap (a kernel's size depends only on kernel size x channels x
  filters, not image size), but each pooling step shrinks the feature map before
  it reaches the Dense layers, which are far more parameter-expensive. The
  3-layer model's smaller final feature map (4x4 vs. 8x8) more than offset the
  cost of its extra conv layer, resulting in fewer total parameters than the
  2-layer model (356,810 vs. 545,098).
- Deeper models took longer to train per epoch (roughly 2-2.5x slower for 3
  layers vs. 1 layer), for diminishing accuracy gains (+4.6 points from 1->2
  layers, +3.9 points from 2->3 layers).

### SageMaker training

The final CNN was also trained on an AWS Academy SageMaker notebook instance to
demonstrate the cloud training workflow, achieving a comparable test accuracy of
approximately 71-73%. The small difference from the local result is expected and
attributable to random weight initialization and data shuffling, neither of which
was fixed with a random seed. Model deployment to a SageMaker endpoint was
attempted but not completed due to AWS Academy Learner Lab account restrictions;
the training and deployment code used is included in this repository as evidence
of the intended workflow.

## Interpretation

**Why did convolutional layers outperform the baseline?**

The CNN outperformed the baseline (73.1% vs. 47.9% test accuracy) despite using
fewer total parameters (356,810 vs. 820,874). This is explained by how each
architecture processes an image. The baseline flattens the image into a single
vector before the network ever sees it, discarding the spatial relationships
between neighboring pixels — it has to relearn any pattern independently for
every possible position, which both wastes parameters and limits what it can
learn. Convolutional layers instead slide a small set of shared weights (a
kernel) across the entire image, so the same edge- or texture-detector is reused
at every position rather than requiring a unique weight per pixel connection.
This weight sharing is precisely why convolution can achieve better performance
with far fewer parameters: a 3x3 kernel's parameter count depends only on its
size and the number of filters, never on the size of the image itself.

**What inductive bias does convolution introduce?**

Convolution introduces two key assumptions about the data. The first is
locality: relevant patterns are assumed to appear within small, continuous
regions of the image, so each kernel only ever looks at a small local
neighborhood at a time. The second is translation equivariance: a learned
pattern (e.g. an edge or a shape) should be recognized the same way regardless of
where it appears in the image, since the same kernel slides across every
position. A horse should be recognized as a horse whether it appears on the left,
right, top, or bottom of the frame.

**In what type of problems would convolution not be appropriate?**

Convolution is not appropriate for data where spatial or sequential adjacency
between values carries no meaning. Tabular data is a common example: the columns
in a table (e.g. age, income, zip code) are typically not spatially related to
one another, and reordering the columns would not change what the data
represents. Applying convolution here would impose an inductive bias — that
nearby values are related and local patterns matter — that does not reflect the
actual structure of the data, and could hurt rather than help learning. Dense
layers, which make no assumption about relationships between input positions,
are generally more appropriate for this kind of data.
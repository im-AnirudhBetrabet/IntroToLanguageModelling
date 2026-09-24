# Character-Level MLP Language Model

This package contains a character-level language model built from scratch using PyTorch tensors.

The implementation follows the progression from Andrej Karpathy's *Neural Networks: Zero to Hero* series, with the goal of understanding the mechanics of neural language models rather than relying on high-level model abstractions.

## Model Architecture

The current model uses a **3-character context window** to predict the next character.

```text
Character Context
       │
       ▼
Character Embeddings
       │
       ▼
Flattened Embeddings
       │
       ▼
Linear Layer
       │
       ▼
Batch Normalization
       │
       ▼
tanh Activation
       │
       ▼
Output Layer
       │
       ▼
Softmax Probabilities
       │
       ▼
Next Character
```

### Model Configuration

| Component           | Configuration |
| ------------------- | ------------: |
| Context window      |  3 characters |
| Embedding dimension |            10 |
| Hidden neurons      |           200 |
| Vocabulary size     |            27 |
| Mini-batch size     |            50 |
| Training iterations |       200,000 |

The vocabulary consists of the 26 English alphabet characters plus `.` as a special start/end token.

## Character Embeddings

Each character is represented using a learned 10-dimensional embedding.

```text
Vocabulary
   ↓
27 × 10 embedding matrix
   ↓
3 characters × 10 dimensions
   ↓
30-dimensional input
```

The embeddings are learned together with the rest of the model during training.

The embedding space can be visualized before and after training using t-SNE to observe how the learned representations change.

## Batch Normalization

Batch Normalization is applied after the hidden linear layer and before the `tanh` activation.

During training, normalization uses the statistics of the current mini-batch.

Running mean and standard deviation are also maintained during training. These running statistics are used during validation, testing, and inference.

The hidden-layer bias is omitted because Batch Normalization provides its own learnable bias and gain parameters.

```text
Linear Layer
     │
     ▼
Batch Normalization
     │
     ▼
tanh
```

## Dataset

The dataset is divided into three subsets:

```text
80% → Training
10% → Validation
10% → Test
```

The training set is used to update the model parameters.

The validation set is used to evaluate the model during development.

The test set is kept separate and used for the final evaluation.

## Training

The model is trained using mini-batch gradient descent.

The training loop performs the following steps:

1. Sample a mini-batch from the training data.
2. Look up character embeddings.
3. Flatten the context embeddings.
4. Apply the hidden linear layer.
5. Apply Batch Normalization.
6. Apply `tanh`.
7. Compute output logits.
8. Calculate cross-entropy loss.
9. Backpropagate the gradients.
10. Update the model parameters.

The learning rate starts at `0.1` and is reduced to `0.01` after 100,000 iterations.

## Parameters

The trainable parameters include:

```text
Embedding weights
Hidden layer weights
BatchNorm gains
BatchNorm biases
Output layer weights
Output layer biases
```

The BatchNorm running mean and running standard deviation are maintained separately and are not trainable parameters.

## Evaluation

The model is evaluated using cross-entropy loss on both the validation and test datasets.

A recent training run produced:

```text
Training loss:    2.0798
Validation loss:  2.1010
Test loss:        2.0992
```

The small difference between the training, validation, and test losses indicates that the model is performing similarly across the available splits for this experiment.

## Text Generation

After training, the model can generate new names character-by-character.

Generation starts with a context containing three `.` tokens:

```text
...
```

The model predicts the next character, samples from the resulting probability distribution, updates the context, and continues until the `.` end token is generated.

Example outputs from one training run:

```text
rashamesrose
licrisa
marissier
tore
jenan
malaite
wan
zain
anna
nasia
eva
ward
```

The generated strings are not required to be existing names. The purpose is to demonstrate that the model has learned character-level patterns from the training data.

## Project Structure

```text
MLP/
├── CharacterMLP.py
├── README.md
└── ...
```

`CharacterMLP.py` contains the complete implementation of the current character-level MLP language model, including:

* Dataset preparation
* Character vocabulary creation
* Character embeddings
* MLP architecture
* Batch Normalization
* Training
* Validation and test evaluation
* Embedding visualization
* Character-level text generation

## Learning Progression

This implementation represents the progression from simpler language models toward neural language models:

```text
Character Frequencies
        ↓
Count-Based Bigram Model
        ↓
Neural Bigram Model
        ↓
Multi-Character Context
        ↓
Learned Character Embeddings
        ↓
Character-Level MLP
        ↓
Batch Normalization
```

The implementation is intentionally kept close to the underlying tensor operations so that the mechanics of the model, forward pass, loss calculation, backpropagation, and parameter updates remain visible.

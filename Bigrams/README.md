# Character-Level Language Models

This directory contains my implementations of **character-level language models**, built while working through Andrej Karpathy's *Neural Networks: Zero to Hero* series.

The implementations progress from a simple count-based bigram model to neural models that learn character representations and use multiple characters as context.

---

## What is a Character-Level Language Model?

A character-level language model learns to predict the next character given some previous context.

For example, given the name:

```text
emma
```

we add special start/end tokens:

```text
.emma.
```

A model can then learn relationships such as:

```text
e → m
m → m
m → a
a → .
```

The goal is to learn these character relationships well enough to generate new names that resemble the patterns found in the training data.

---

# Implementations

## 1. Count-Based Bigram Model

**File:** `bigram_counts.py`

The first implementation builds a language model directly from bigram frequencies.

A **bigram** consists of two consecutive characters:

```text
(current character, next character)
```

The process is:

```text
Names
  ↓
Character pairs
  ↓
Bigram counts
  ↓
Probability distribution
  ↓
Sample new names
```

The implementation:

* Builds a vocabulary containing the 26 letters and a `.` start/end token.
* Counts occurrences of every character pair.
* Uses Laplace smoothing to avoid zero probabilities.
* Converts counts into probability distributions.
* Generates new names by sampling from the learned distributions.
* Evaluates the model using:

  * Log Likelihood
  * Negative Log Likelihood
  * Average Negative Log Likelihood

This provides a simple, non-neural baseline for character-level language modelling.

---

## 2. Neural Bigram Model

**File:** `bigram_model.py`

The second implementation recreates the bigram model using a simple neural network.

Instead of directly storing the character transition probabilities, the model learns a weight matrix that represents the relationships between consecutive characters.

The process is:

```text
Character
    ↓
One-Hot Encoding
    ↓
Weight Matrix
    ↓
Logits
    ↓
Softmax
    ↓
Probability Distribution
    ↓
Loss
    ↓
Backpropagation
    ↓
Gradient Descent
```

The implementation:

* Represents characters using one-hot encoding.
* Uses a `27 × 27` weight matrix.
* Computes logits from the one-hot encoded input.
* Converts logits into probabilities using softmax.
* Uses Negative Log Likelihood with L2 regularization..
* Adds L2 regularization.
* Trains the weights using gradient descent.
* Generates new names by sampling from the learned probability distribution.

This demonstrates that the count-based bigram model can also be expressed as a neural model.

---

## 3. Multi-Character Context MLP

**File:** `bigram_mlp.py`

The third implementation extends the previous models by using **multiple previous characters as context** instead of only one.

The model uses a fixed context window of three characters.

For example:

```text
... → e
..e → m
.em → m
emm → a
mma → .
```

The architecture is:

```text
Character indices
       ↓
Character embeddings
       ↓
3-character context
       ↓
Flatten
       ↓
Hidden layer
       ↓
tanh activation
       ↓
Output layer
       ↓
27 character probabilities
```

Each character is represented using a learnable embedding.

For example, with an embedding size of 10:

```text
27 characters × 10 dimensions
```

A three-character context therefore produces:

```text
3 × 10 = 30 values
```

which are passed into the first layer of the MLP.

The implementation:

* Uses a 3-character context window.
* Learns character embeddings.
* Uses a hidden layer with 200 neurons.
* Uses `tanh` as the activation function.
* Predicts one of 27 possible characters.
* Uses cross-entropy loss.
* Trains using mini-batch gradient descent.
* Splits the dataset into training, validation, and test sets.
* Evaluates the model on validation and test data.
* Generates names by sampling from the learned probability distribution.
* Visualizes the learned character embeddings before and after training.

---

# Character Embeddings

The MLP introduces a learnable embedding matrix:

```text
27 × 10
```

The 27 rows correspond to the 27 possible characters:

```text
. a b c ... z
```

The 10 columns represent the learned dimensions of each character's embedding.

These embeddings are **learned during training**, just like the other model parameters.

The embeddings can also be visualized to observe how the character representations change during training.

The visualization uses t-SNE to project the higher-dimensional embeddings into two dimensions.

---

# Training

The MLP is trained using mini-batches.

For each iteration:

```text
Sample a mini-batch
       ↓
Embedding lookup
       ↓
Hidden layer
       ↓
tanh
       ↓
Output logits
       ↓
Cross-entropy loss
       ↓
Backpropagation
       ↓
Update model parameters
```

The model parameters include:

```text
C   → Character embeddings
W1  → First layer weights
b1  → First layer biases
W2  → Output layer weights
b2  → Output layer biases
```

Gradients are cleared before each backward pass so that gradients from previous iterations do not accumulate.

---

# Dataset Splits

The dataset is shuffled and divided into three parts:

```text
80% → Training set
10% → Validation set
10% → Test set
```

The training set is used to update the model parameters.

The validation set can be used to evaluate the model during development and experimentation.

The test set is kept separate for the final evaluation.

---

# Evaluation

The models use **Negative Log Likelihood** or **Cross Entropy** to measure how well they predict the observed next characters.

For a predicted probability `p`:

```text
Negative Log Likelihood = -log(p)
```

Lower loss means the model is assigning higher probability to the correct next character.

---

# Files

```text
Bigrams/
├── README.md
├── bigram_counts.py
├── bigram_model.py
├── bigram_mlp.py
├── bigram_mlp_exploration.py
├── __init__.py
└── utils/
    ├── __init__.py
    ├── embedding_plotter.py
    └── make_dataset.py
```

---

# Key Concepts

This stage covers:

* Character-level language modelling
* Vocabulary creation
* Bigram statistics
* Probability distributions
* Laplace smoothing
* Sampling
* One-hot encoding
* Character embeddings
* Multi-character context
* Weight matrices
* Logits
* Softmax
* Cross-entropy
* Negative Log Likelihood
* L2 regularization
* `tanh` activation
* Backpropagation
* Gradient descent
* Mini-batch training
* Training / validation / test splits
* Embedding visualization

---

# The Progression

The main goal of these implementations is to understand how increasingly capable language models can be built from simple components.

```text
Count-based Bigram
       ↓
Neural Bigram
       ↓
Multi-Character Context
       ↓
Character Embeddings
       ↓
MLP
       ↓
Neural Language Model
```

The important transition is from simply **counting character relationships** to **learning representations and parameters that can model those relationships**.

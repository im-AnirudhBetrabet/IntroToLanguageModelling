# Character-Level Bigram Language Models

This directory contains my implementations of a **character-level Bigram Language Model**, built while working through Andrej Karpathy's *Neural Networks: Zero to Hero* series.

The goal of this stage is to understand how a simple language model can learn the relationships between consecutive characters, first using **counts and probabilities**, and then using a **neural network**.

---

## What is a Bigram?

A **bigram** is a pair of consecutive elements.

For example, given the name:

```text
emma
```

we add special start/end tokens:

```text
.e m m a.
```

This produces the bigrams:

```text
(. , e)
(e , m)
(m , m)
(m , a)
(a , .)
```

The model learns the probability of the next character given the current character:

```text
P(next character | current character)
```

---

## Implementations

### 1. Count-Based Bigram Model

**File:** `bigram_counts.py`

The first implementation builds the language model directly from character frequencies.

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

---

### 2. Neural Bigram Model

**File:** `bigram_model.py`

The second implementation recreates the bigram model using a simple neural network.

Instead of directly storing the probabilities, the model learns a **weight matrix**.

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

* Encodes characters using one-hot vectors.
* Uses a `27 × 27` weight matrix.
* Computes logits from the one-hot encoded input.
* Converts logits into probabilities using softmax.
* Uses Negative Log Likelihood as the main loss.
* Adds L2 regularization to the loss.
* Trains the weights using gradient descent.
* Generates new names by sampling from the learned probability distribution.

---

## Evaluation

Both implementations use **Negative Log Likelihood (NLL)** to evaluate how well the model predicts the observed character transitions.

For a predicted probability `p`:

```text
Negative Log Likelihood = -log(p)
```

A lower average NLL indicates that the model assigns higher probability to the character transitions present in the dataset.

---

## Dataset

The models are trained on `names.txt`, which contains a collection of names.

The special character:

```text
.
```

is used as both the **start-of-word** and **end-of-word** token.

This allows the model to learn both:

```text
. → first character
```

and

```text
last character → .
```

---

## Key Concepts

This stage covers the following concepts:

* Character-level language modelling
* Vocabulary creation
* Bigram statistics
* Probability distributions
* Laplace smoothing
* Sampling
* One-hot encoding
* Weight matrices
* Logits
* Softmax
* Negative Log Likelihood
* L2 regularization
* Gradient descent
* Backpropagation

---

## The Main Idea

The interesting part of this stage is seeing the same basic language-modeling idea implemented in two different ways:

### Count-based

```text
How often does character B follow character A?
```

### Neural

```text
Can a weight matrix learn those relationships?
```

The neural implementation shows how the probability-based model can be expressed as a simple neural network and optimized using gradient descent.

---

## Files

```text
Bigrams/
├── README.md
├── bigram_counts.py
├── bigram_model.py
└── __init__.py
```

This directory represents the first step from **count-based language modelling toward neural language models**.

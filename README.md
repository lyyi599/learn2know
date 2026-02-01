# Sutskever 30 - NumPy + PyTorch Learning Suite

**Freshman-friendly, step-by-step implementations of the 30 foundational papers recommended by Ilya Sutskever**

[![Implementations](https://img.shields.io/badge/Implementations-NumPy%2030%2F30-brightgreen)](https://github.com/pageman/sutskever-30-implementations)
[![Coverage](https://img.shields.io/badge/Track-PyTorch%20in%20progress-orange)](https://github.com/pageman/sutskever-30-implementations)
[![Python](https://img.shields.io/badge/Python-NumPy%20%2B%20PyTorch-yellow)](https://numpy.org/)

## Overview

This repository is a fork of the original NumPy-only implementation suite and extends it with a PyTorch learning track. The goal is to help freshmen learn core ideas twice: first from scratch with NumPy, then with the practical PyTorch APIs.

? **Original repository:** [pageman/sutskever-30-implementations](https://github.com/pageman/sutskever-30-implementations)

Status:
- NumPy track: 30/30 notebooks (upstream complete).
- PyTorch track: planned and in progress (see `docs/kk.md` for the implementation plan).

Each implementation:
- Uses NumPy for educational clarity
- Includes synthetic/bootstrapped data for immediate execution
- Provides visualizations and explanations
- Demonstrates core concepts from each paper
- Runs in Jupyter notebooks for interactive learning

## Freshman Tutorial Goals

This repo is designed for beginners who want to understand how deep learning works under the hood.

You will:
- Build intuition with NumPy first (explicit math and arrays).
- Rebuild the same ideas in PyTorch (practical engineering).
- Compare API-level abstractions to raw operations.

Recommended learning order:
1. Read notebooks/00_numpy_and_pytorch_for_deep_learning.ipynb to get the basics.
2. Complete a NumPy notebook first.
3. Implement its PyTorch version next.

## Quick Start

```bash
# Navigate to the directory
cd sutskever-30-implementations

# Install dependencies
pip install numpy matplotlib scipy
# We recommend setting up a virtual environment before diving into PyTorch. This helps keep your global Python environment clean and prevents dependency conflicts.
# PyTorch stack
pip install torch torchvision torchaudio

# Run any notebook
jupyter notebook notebooks/02_char_rnn_karpathy.ipynb
```

## The Sutskever 30 Papers (NumPy Track)

### Foundational Concepts (Papers 1-5)

| # | Paper | Notebook | Key Concepts |
|---|-------|----------|--------------|
| 1 | The First Law of Complexodynamics | `notebooks/01_complexity_dynamics.ipynb` | Entropy, Complexity Growth, Cellular Automata |
| 2 | The Unreasonable Effectiveness of RNNs | `notebooks/02_char_rnn_karpathy.ipynb` | Character-level models, RNN basics, Text generation |
| 3 | Understanding LSTM Networks | `notebooks/03_lstm_understanding.ipynb` | Gates, Long-term memory, Gradient flow |
| 4 | RNN Regularization | `notebooks/04_rnn_regularization.ipynb` | Dropout for sequences, Variational dropout |
| 5 | Keeping Neural Networks Simple | `notebooks/05_neural_network_pruning.ipynb` | MDL principle, Weight pruning, 90%+ sparsity |

### Architectures & Mechanisms (Papers 6-15)

| # | Paper | Notebook | Key Concepts |
|---|-------|----------|--------------|
| 6 | Pointer Networks | `notebooks/06_pointer_networks.ipynb` | Attention as pointer, Combinatorial problems |
| 7 | ImageNet/AlexNet | `notebooks/07_alexnet_cnn.ipynb` | CNNs, Convolution, Data augmentation |
| 8 | Order Matters: Seq2Seq for Sets | `notebooks/08_seq2seq_for_sets.ipynb` | Set encoding, Permutation invariance, Attention pooling |
| 9 | GPipe | `notebooks/09_gpipe.ipynb` | Pipeline parallelism, Micro-batching, Re-materialization |
| 10 | Deep Residual Learning (ResNet) | `notebooks/10_resnet_deep_residual.ipynb` | Skip connections, Gradient highways |
| 11 | Dilated Convolutions | `notebooks/11_dilated_convolutions.ipynb` | Receptive fields, Multi-scale |
| 12 | Neural Message Passing (GNNs) | `notebooks/12_graph_neural_networks.ipynb` | Graph networks, Message passing |
| 13 | **Attention Is All You Need** | `notebooks/13_attention_is_all_you_need.ipynb` | Transformers, Self-attention, Multi-head |
| 14 | Neural Machine Translation | `notebooks/14_bahdanau_attention.ipynb` | Seq2seq, Bahdanau attention |
| 15 | Identity Mappings in ResNet | `notebooks/15_identity_mappings_resnet.ipynb` | Pre-activation, Gradient flow |

### Advanced Topics (Papers 16-22)

| # | Paper | Notebook | Key Concepts |
|---|-------|----------|--------------|
| 16 | Relational Reasoning | `notebooks/16_relational_reasoning.ipynb` | Relation networks, Pairwise functions |
| 17 | **Variational Lossy Autoencoder** | `notebooks/17_variational_autoencoder.ipynb` | VAE, ELBO, Reparameterization trick |
| 18 | **Relational RNNs** | `notebooks/18_relational_rnn.ipynb` | Relational memory, Multi-head self-attention, Manual backprop (~1100 lines) |
| 19 | The Coffee Automaton | `notebooks/19_coffee_automaton.ipynb` | Irreversibility, Entropy, Arrow of time, Landauer's principle |
| 20 | **Neural Turing Machines** | `notebooks/20_neural_turing_machine.ipynb` | External memory, Differentiable addressing |
| 21 | Deep Speech 2 (CTC) | `notebooks/21_ctc_speech.ipynb` | CTC loss, Speech recognition |
| 22 | **Scaling Laws** | `notebooks/22_scaling_laws.ipynb` | Power laws, Compute-optimal training |

### Theory & Meta-Learning (Papers 23-30)

| # | Paper | Notebook | Key Concepts |
|---|-------|----------|--------------|
| 23 | MDL Principle | `notebooks/23_mdl_principle.ipynb` | Information theory, Model selection, Compression |
| 24 | **Machine Super Intelligence** | `notebooks/24_machine_super_intelligence.ipynb` | Universal AI, AIXI, Solomonoff induction, Intelligence measures, Self-improvement |
| 25 | Kolmogorov Complexity | `notebooks/25_kolmogorov_complexity.ipynb` | Compression, Algorithmic randomness, Universal prior |
| 26 | **CS231n: CNNs for Visual Recognition** | `notebooks/26_cs231n_cnn_fundamentals.ipynb` | Image classification pipeline, kNN/Linear/NN/CNN, Backprop, Optimization, Babysitting neural nets |
| 27 | Multi-token Prediction | `notebooks/27_multi_token_prediction.ipynb` | Multiple future tokens, Sample efficiency, 2-3x faster |
| 28 | Dense Passage Retrieval | `notebooks/28_dense_passage_retrieval.ipynb` | Dual encoders, MIPS, In-batch negatives |
| 29 | Retrieval-Augmented Generation | `notebooks/29_rag.ipynb` | RAG-Sequence, RAG-Token, Knowledge retrieval |
| 30 | Lost in the Middle | `notebooks/30_lost_in_middle.ipynb` | Position bias, Long context, U-shaped curve |

## Featured Implementations

### Must-Read Notebooks

These implementations cover the most influential papers and demonstrate core deep learning concepts:

#### Foundations
1. **`notebooks/02_char_rnn_karpathy.ipynb`** - Character-level RNN
  - Build RNN from scratch
  - Understand backpropagation through time
  - Generate text

2. **`notebooks/03_lstm_understanding.ipynb`** - LSTM Networks
  - Implement forget/input/output gates
  - Visualize gate activations
  - Compare with vanilla RNN

3. **`notebooks/04_rnn_regularization.ipynb`** - RNN Regularization
  - Variational dropout for RNNs
  - Proper dropout placement
  - Training improvements

4. **`notebooks/05_neural_network_pruning.ipynb`** - Network Pruning & MDL
  - Magnitude-based pruning
  - Iterative pruning with fine-tuning
  - 90%+ sparsity with minimal loss
  - Minimum Description Length principle

#### Computer Vision
5. **`notebooks/07_alexnet_cnn.ipynb`** - CNNs & AlexNet
  - Convolutional layers from scratch
  - Max pooling and ReLU
  - Data augmentation techniques

6. **`notebooks/10_resnet_deep_residual.ipynb`** - ResNet
  - Skip connections solve degradation
  - Gradient flow visualization
  - Identity mapping intuition

7. **`notebooks/15_identity_mappings_resnet.ipynb`** - Pre-activation ResNet
  - Pre-activation vs post-activation
  - Better gradient flow
  - Training 1000+ layer networks

8. **`notebooks/11_dilated_convolutions.ipynb`** - Dilated Convolutions
  - Multi-scale receptive fields
  - No pooling required
  - Semantic segmentation

#### Attention & Transformers
9. **`notebooks/14_bahdanau_attention.ipynb`** - Neural Machine Translation
  - Original attention mechanism
  - Seq2seq with alignment
  - Attention visualization

10. **`notebooks/13_attention_is_all_you_need.ipynb`** - Transformers
  - Scaled dot-product attention
  - Multi-head attention
  - Positional encoding
  - Foundation of modern LLMs

11. **`notebooks/06_pointer_networks.ipynb`** - Pointer Networks
  - Attention as selection
  - Combinatorial optimization
  - Variable output size

12. **`notebooks/08_seq2seq_for_sets.ipynb`** - Seq2Seq for Sets
  - Permutation-invariant set encoder
  - Read-Process-Write architecture
  - Attention over unordered elements
  - Sorting and set operations
  - Comparison: order-sensitive vs order-invariant

13. **`notebooks/09_gpipe.ipynb`** - GPipe Pipeline Parallelism
  - Model partitioning across devices
  - Micro-batching for pipeline utilization
  - F-then-B schedule (forward all, backward all)
  - Re-materialization (gradient checkpointing)
  - Bubble time analysis
  - Training models larger than single-device memory

#### Advanced Topics
14. **`notebooks/12_graph_neural_networks.ipynb`** - Graph Neural Networks
  - Message passing framework
  - Graph convolutions
  - Molecular property prediction

15. **`notebooks/16_relational_reasoning.ipynb`** - Relation Networks
  - Pairwise relational reasoning
  - Visual QA
  - Permutation invariance

16. **`notebooks/18_relational_rnn.ipynb`** - Relational RNN
  - LSTM with relational memory
  - Multi-head self-attention across memory slots
  - Architecture demonstration (forward pass)
  - Sequential reasoning tasks
  - **Section 11: Manual backpropagation implementation (~1100 lines)**
  - Complete gradient computation for all components
  - Gradient checking with numerical verification

17. **`notebooks/20_neural_turing_machine.ipynb`** - Memory-Augmented Networks
  - Content & location addressing
  - Differentiable read/write
  - External memory

18. **`notebooks/21_ctc_speech.ipynb`** - CTC Loss & Speech Recognition
  - Connectionist Temporal Classification
  - Alignment-free training
  - Forward algorithm

#### Generative Models
19. **`notebooks/17_variational_autoencoder.ipynb`** - VAE
  - Generative modeling
  - ELBO loss
  - Latent space visualization

#### Modern Applications
20. **`notebooks/27_multi_token_prediction.ipynb`** - Multi-Token Prediction
  - Predict multiple future tokens
  - 2-3x sample efficiency
  - Speculative decoding
  - Faster training & inference

21. **`notebooks/28_dense_passage_retrieval.ipynb`** - Dense Retrieval
  - Dual encoder architecture
  - In-batch negatives
  - Semantic search

22. **`notebooks/29_rag.ipynb`** - Retrieval-Augmented Generation
  - RAG-Sequence vs RAG-Token
  - Combining retrieval + generation
  - Knowledge-grounded outputs

23. **`notebooks/30_lost_in_middle.ipynb`** - Long Context Analysis
  - Position bias in LLMs
  - U-shaped performance curve
  - Document ordering strategies

#### Scaling & Theory
24. **`notebooks/22_scaling_laws.ipynb`** - Scaling Laws
  - Power law relationships
  - Compute-optimal training
  - Performance prediction

25. **`notebooks/23_mdl_principle.ipynb`** - Minimum Description Length
  - Information-theoretic model selection
  - Compression = Understanding
  - MDL vs AIC/BIC comparison
  - Neural network architecture selection
  - MDL-based pruning (connects to Paper 5)
  - Kolmogorov complexity preview

26. **`notebooks/25_kolmogorov_complexity.ipynb`** - Kolmogorov Complexity
  - K(x) = shortest program generating x
  - Randomness = Incompressibility
  - Algorithmic probability (Solomonoff)
  - Universal prior for induction
  - Connection to Shannon entropy
  - Occam's Razor formalized
  - Theoretical foundation for ML

27. **`notebooks/24_machine_super_intelligence.ipynb`** - Universal Artificial Intelligence
  - **Formal theory of intelligence (Legg & Hutter)**
  - Psychometric g-factor and universal intelligence Y
  - Solomonoff induction for sequence prediction
  - AIXI: Theoretically optimal RL agent
  - Monte Carlo AIXI (MC-AIXI) approximation
  - Kolmogorov complexity estimation
  - Intelligence measurement across environments
  - Recursive self-improvement dynamics
  - Intelligence explosion scenarios
  - **6 sections: from psychometrics to superintelligence**
  - Connects Papers #23 (MDL), #25 (Kolmogorov), #8 (DQN)

28. **`notebooks/01_complexity_dynamics.ipynb`** - Complexity & Entropy
  - Cellular automata (Rule 30)
  - Entropy growth
  - Irreversibility (basic introduction)

28. **`notebooks/19_coffee_automaton.ipynb`** - The Coffee Automaton (Deep Dive)
  - **Comprehensive exploration of irreversibility**
  - Coffee mixing and diffusion processes
  - Entropy growth and coarse-graining
  - Phase space and Liouville's theorem
  - Poincare recurrence theorem (will unmix after e^N time!)
  - Maxwell's demon and Landauer's principle
  - Computational irreversibility (one-way functions, hashing)
  - Information bottleneck in machine learning
  - Biological irreversibility (life and the 2nd law)
  - Arrow of time: fundamental vs emergent
  - **10 comprehensive sections exploring irreversibility across all scales**

29. **`notebooks/26_cs231n_cnn_fundamentals.ipynb`** - CS231n: Vision from First Principles
  - **Complete vision pipeline in pure NumPy**
  - k-Nearest Neighbors baseline
  - Linear classifiers (SVM and Softmax)
  - Optimization (SGD, Momentum, Adam, learning rate schedules)
  - 2-layer neural networks with backpropagation
  - Convolutional layers (conv, pool, ReLU)
  - Complete CNN architecture (Mini-AlexNet)
  - Visualization techniques (filters, saliency maps)
  - Transfer learning principles
  - Babysitting tips (sanity checks, hyperparameter tuning, monitoring)
  - **10 sections covering entire CS231n curriculum**
  - Ties together Papers #7 (AlexNet), #10 (ResNet), #11 (Dilated Conv)

## Repository Structure

See `STRUCTURE.md` for the up-to-date repository layout.


## Learning Path

### Beginner Track (Start here!)
1. **Character RNN** (`notebooks/02_char_rnn_karpathy.ipynb`) - Learn basic RNNs
2. **LSTM** (`notebooks/03_lstm_understanding.ipynb`) - Understand gating mechanisms
3. **CNNs** (`notebooks/07_alexnet_cnn.ipynb`) - Computer vision fundamentals
4. **ResNet** (`notebooks/10_resnet_deep_residual.ipynb`) - Skip connections
5. **VAE** (`notebooks/17_variational_autoencoder.ipynb`) - Generative models

### Intermediate Track
6. **RNN Regularization** (`notebooks/04_rnn_regularization.ipynb`) - Better training
7. **Bahdanau Attention** (`notebooks/14_bahdanau_attention.ipynb`) - Attention basics
8. **Pointer Networks** (`notebooks/06_pointer_networks.ipynb`) - Attention as selection
9. **Seq2Seq for Sets** (`notebooks/08_seq2seq_for_sets.ipynb`) - Permutation invariance
10. **CS231n** (`notebooks/26_cs231n_cnn_fundamentals.ipynb`) - Complete vision pipeline (kNN ->CNNs)
11. **GPipe** (`notebooks/09_gpipe.ipynb`) - Pipeline parallelism for large models
12. **Transformers** (`notebooks/13_attention_is_all_you_need.ipynb`) - Modern architecture
13. **Dilated Convolutions** (`notebooks/11_dilated_convolutions.ipynb`) - Receptive fields
14. **Scaling Laws** (`notebooks/22_scaling_laws.ipynb`) - Understanding scale

### Advanced Track
15. **Pre-activation ResNet** (`notebooks/15_identity_mappings_resnet.ipynb`) - Architecture details
16. **Graph Neural Networks** (`notebooks/12_graph_neural_networks.ipynb`) - Graph learning
17. **Relation Networks** (`notebooks/16_relational_reasoning.ipynb`) - Relational reasoning
18. **Neural Turing Machines** (`notebooks/20_neural_turing_machine.ipynb`) - External memory
19. **CTC Loss** (`notebooks/21_ctc_speech.ipynb`) - Speech recognition
20. **Dense Retrieval** (`notebooks/28_dense_passage_retrieval.ipynb`) - Semantic search
21. **RAG** (`notebooks/29_rag.ipynb`) - Retrieval-augmented generation
22. **Lost in the Middle** (`notebooks/30_lost_in_middle.ipynb`) - Long context analysis

### Theory & Fundamentals
23. **MDL Principle** (`notebooks/23_mdl_principle.ipynb`) - Model selection via compression
24. **Kolmogorov Complexity** (`notebooks/25_kolmogorov_complexity.ipynb`) - Randomness & information
25. **Complexity Dynamics** (`notebooks/01_complexity_dynamics.ipynb`) - Entropy & emergence
26. **Coffee Automaton** (`notebooks/19_coffee_automaton.ipynb`) - Deep dive into irreversibility

## Key Insights from the Sutskever 30

### Architecture Evolution
- **RNN ->LSTM**: Gating solves vanishing gradients
- **Plain Networks ->ResNet**: Skip connections enable depth
- **RNN ->Transformer**: Attention enables parallelization
- **Fixed vocab ->Pointers**: Output can reference input

### Fundamental Mechanisms
- **Attention**: Differentiable selection mechanism
- **Residual Connections**: Gradient highways
- **Gating**: Learned information flow control
- **External Memory**: Separate storage from computation

### Training Insights
- **Scaling Laws**: Performance predictably improves with scale
- **Regularization**: Dropout, weight decay, data augmentation
- **Optimization**: Gradient clipping, learning rate schedules
- **Compute-Optimal**: Balance model size and training data

### Theoretical Foundations
- **Information Theory**: Compression, entropy, MDL
- **Complexity**: Kolmogorov complexity, power laws
- **Generative Modeling**: VAE, ELBO, latent spaces
- **Memory**: Differentiable data structures

## Implementation Philosophy

### Why NumPy-only?

These implementations deliberately avoid PyTorch/TensorFlow to:
- **Deepen understanding**: See what frameworks abstract away
- **Educational clarity**: No magic, every operation explicit
- **Core concepts**: Focus on algorithms, not framework APIs
- **Transferable knowledge**: Principles apply to any framework

### Synthetic Data Approach

Each notebook generates its own data to:
- **Immediate execution**: No dataset downloads required
- **Controlled experiments**: Understand behavior on simple cases
- **Concept focus**: Data doesn't obscure the algorithm
- **Rapid iteration**: Modify and re-run instantly

## Extensions & Next Steps

### Build on These Implementations

After understanding the core concepts, try:

1. **Scale up**: Implement in PyTorch/JAX for real datasets
2. **Combine techniques**: E.g., ResNet + Attention
3. **Modern variants**:
  - RNN ->GRU ->Transformer
  - VAE ->beta-VAE ->VQ-VAE
  - ResNet ->ResNeXt ->EfficientNet
4. **Applications**: Apply to real problems

### Research Directions

The Sutskever 30 points toward:
- Scaling (bigger models, more data)
- Efficiency (sparse models, quantization)
- Capabilities (reasoning, multi-modal)
- Understanding (interpretability, theory)

## Resources

### Original Papers
See `docs/IMPLEMENTATION_TRACKS.md` for full citations and links

### Additional Reading
- [Ilya Sutskever's Reading List (GitHub)](https://github.com/dzyim/ilya-sutskever-recommended-reading)
- [Aman's AI Journal - Sutskever 30 Primers](https://aman.ai/primers/ai/top-30-papers/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Andrej Karpathy's Blog](http://karpathy.github.io/)

### Courses
- Stanford CS231n: Convolutional Neural Networks
- Stanford CS224n: NLP with Deep Learning
- MIT 6.S191: Introduction to Deep Learning

## Contributing

These implementations are educational and can be improved! Consider:
- Adding more visualizations
- Implementing missing papers
- Improving explanations
- Finding bugs
- Adding comparisons with framework implementations

## Citation

If you use these implementations in your work or teaching:

```bibtex
@misc{sutskever30implementations,
 title={Sutskever 30: Complete Implementation Suite},
 author={Paul "The Pageman" Pajo, pageman@gmail.com},
 year={2025},
 note={Educational implementations of Ilya Sutskever's recommended reading list, inspired by https://papercode.vercel.app/}
}
```

## License

Educational use. See individual papers for original research citations.

## Acknowledgments

- **Ilya Sutskever**: For curating this essential reading list
- **Paper authors**: For their foundational contributions
- **Community**: For making these ideas accessible

---

## Latest Additions (December 2025)

### Recently Implemented (21 new papers!)
- **Paper 4**: RNN Regularization (variational dropout)
- **Paper 5**: Neural Network Pruning (MDL, 90%+ sparsity)
- **Paper 7**: AlexNet (CNNs from scratch)
- **Paper 8**: Seq2Seq for Sets (permutation invariance, attention pooling)
- **Paper 9**: GPipe (pipeline parallelism, micro-batching, re-materialization)
- **Paper 19**: The Coffee Automaton (deep dive into irreversibility, entropy, Landauer's principle)
- **Paper 26**: CS231n (complete vision pipeline: kNN ->CNN, all in NumPy)
- **Paper 11**: Dilated Convolutions (multi-scale)
- **Paper 12**: Graph Neural Networks (message passing)
- **Paper 14**: Bahdanau Attention (original attention)
- **Paper 15**: Identity Mappings ResNet (pre-activation)
- **Paper 16**: Relational Reasoning (relation networks)
- **Paper 18**: Relational RNNs (relational memory + Section 11: manual backprop ~1100 lines)
- **Paper 21**: Deep Speech 2 (CTC loss)
- **Paper 23**: MDL Principle (compression, model selection, connects to Papers 5 & 25)
- **Paper 24**: Machine Super Intelligence (Universal AI, AIXI, Solomonoff induction, intelligence measures, recursive self-improvement)
- **Paper 25**: Kolmogorov Complexity (randomness, algorithmic probability, theoretical foundation)
- **Paper 27**: Multi-Token Prediction (2-3x sample efficiency)
- **Paper 28**: Dense Passage Retrieval (dual encoders)
- **Paper 29**: RAG (retrieval-augmented generation)
- **Paper 30**: Lost in the Middle (long context)

## Quick Reference: Implementation Complexity

### Can Implement in an Afternoon
- Character RNN
- LSTM
- ResNet
- Simple VAE
- Dilated Convolutions

### Weekend Projects
- Transformer
- Pointer Networks
- Graph Neural Networks
- Relation Networks
- Neural Turing Machine
- CTC Loss
- Dense Retrieval

### Week-Long Deep Dives
- Full RAG system
-  Large-scale experiments
-  Hyperparameter optimization

---

**"If you really learn all of these, you'll know 90% of what matters today."** - Ilya Sutskever

Happy learning! 












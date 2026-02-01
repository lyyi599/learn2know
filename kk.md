# PyTorch Implementation Plan (Fork of Sutskever 30)

## Purpose
Create PyTorch versions of the 30 NumPy notebooks in this repository, optimized for freshmen learning. Each PyTorch notebook should mirror the NumPy version conceptually while teaching practical PyTorch usage.

## Scope
- Preserve the existing NumPy notebooks (source of truth for math and explanations).
- Add PyTorch notebooks with consistent naming and structure.
- Keep datasets synthetic or lightweight to ensure quick execution.
- Emphasize clarity over performance; prefer simple code to advanced tricks.

## Naming and Structure
- NumPy notebooks remain: `01_complexity_dynamics.ipynb`, ..., `30_lost_in_middle.ipynb`.
- PyTorch notebooks will be added as: `pt_01_complexity_dynamics.ipynb`, ..., `pt_30_lost_in_middle.ipynb`.
- Optional shared utilities (if needed): `pt_utils.py` for small helper functions.

## Notebook Template (PyTorch)
Each PyTorch notebook should follow this consistent layout:
1. Title and learning goals (2-5 bullets).
2. Short recap of the NumPy version (what to reuse conceptually).
3. Data generation (synthetic or minimal).
4. Model definition (torch.nn.Module).
5. Training loop (forward, loss, backward, step).
6. Evaluation and visualization (matplotlib).
7. Compare NumPy vs PyTorch (what changes, what stays the same).
8. Exercises (2-4 small tasks for students).

## Quality Bar
- Runs on CPU in a few minutes.
- Clear variable names and minimal magic.
- Explanations focused on first-principles and PyTorch APIs.
- Figures replicate the insights from the NumPy notebook.

## Implementation Phases

### Phase 0: Setup and Template
- Create a PyTorch notebook template with the structure above.
- Add a short PyTorch quickstart cell (device, seed, tensor basics).
- Validate the template by converting the simplest notebook.

### Phase 1: Fundamentals (Papers 1-5)
- pt_01_complexity_dynamics.ipynb
- pt_02_char_rnn_karpathy.ipynb
- pt_03_lstm_understanding.ipynb
- pt_04_rnn_regularization.ipynb
- pt_05_neural_network_pruning.ipynb
Deliverable: one-by-one conversion with strict parity to NumPy explanations.

### Phase 2: Core Architectures (Papers 6-15)
- pt_06_pointer_networks.ipynb
- pt_07_alexnet_cnn.ipynb
- pt_08_seq2seq_for_sets.ipynb
- pt_09_gpipe.ipynb
- pt_10_resnet_deep_residual.ipynb
- pt_11_dilated_convolutions.ipynb
- pt_12_graph_neural_networks.ipynb
- pt_13_attention_is_all_you_need.ipynb
- pt_14_bahdanau_attention.ipynb
- pt_15_identity_mappings_resnet.ipynb
Deliverable: keep models small and focus on core mechanisms.

### Phase 3: Advanced Topics (Papers 16-22)
- pt_16_relational_reasoning.ipynb
- pt_17_variational_autoencoder.ipynb
- pt_18_relational_rnn.ipynb
- pt_19_coffee_automaton.ipynb
- pt_20_neural_turing_machine.ipynb
- pt_21_ctc_speech.ipynb
- pt_22_scaling_laws.ipynb
Deliverable: replicate key graphs and qualitative behaviors.

### Phase 4: Theory and Modern Applications (Papers 23-30)
- pt_23_mdl_principle.ipynb
- pt_24_machine_super_intelligence.ipynb
- pt_25_kolmogorov_complexity.ipynb
- pt_26_cs231n_cnn_fundamentals.ipynb
- pt_27_multi_token_prediction.ipynb
- pt_28_dense_passage_retrieval.ipynb
- pt_29_rag.ipynb
- pt_30_lost_in_middle.ipynb
Deliverable: focus on teaching ideas with toy or simplified PyTorch versions.

## Per-Notebook Checklist
- Parity: match the NumPy notebook sections and math.
- Minimal data: synthetic or tiny sample data.
- Core model: implement only the essential components.
- Clear training loop: show the gradients and updates.
- Visualization: reproduce at least one key plot.
- Exercises: include 2-4 short tasks.

## Tracking Table (Initial)

| # | NumPy Notebook | PyTorch Notebook | Status |
|---|---------------|------------------|--------|
| 01 | 01_complexity_dynamics.ipynb | pt_01_complexity_dynamics.ipynb | todo |
| 02 | 02_char_rnn_karpathy.ipynb | pt_02_char_rnn_karpathy.ipynb | todo |
| 03 | 03_lstm_understanding.ipynb | pt_03_lstm_understanding.ipynb | todo |
| 04 | 04_rnn_regularization.ipynb | pt_04_rnn_regularization.ipynb | todo |
| 05 | 05_neural_network_pruning.ipynb | pt_05_neural_network_pruning.ipynb | todo |
| 06 | 06_pointer_networks.ipynb | pt_06_pointer_networks.ipynb | todo |
| 07 | 07_alexnet_cnn.ipynb | pt_07_alexnet_cnn.ipynb | todo |
| 08 | 08_seq2seq_for_sets.ipynb | pt_08_seq2seq_for_sets.ipynb | todo |
| 09 | 09_gpipe.ipynb | pt_09_gpipe.ipynb | todo |
| 10 | 10_resnet_deep_residual.ipynb | pt_10_resnet_deep_residual.ipynb | todo |
| 11 | 11_dilated_convolutions.ipynb | pt_11_dilated_convolutions.ipynb | todo |
| 12 | 12_graph_neural_networks.ipynb | pt_12_graph_neural_networks.ipynb | todo |
| 13 | 13_attention_is_all_you_need.ipynb | pt_13_attention_is_all_you_need.ipynb | todo |
| 14 | 14_bahdanau_attention.ipynb | pt_14_bahdanau_attention.ipynb | todo |
| 15 | 15_identity_mappings_resnet.ipynb | pt_15_identity_mappings_resnet.ipynb | todo |
| 16 | 16_relational_reasoning.ipynb | pt_16_relational_reasoning.ipynb | todo |
| 17 | 17_variational_autoencoder.ipynb | pt_17_variational_autoencoder.ipynb | todo |
| 18 | 18_relational_rnn.ipynb | pt_18_relational_rnn.ipynb | todo |
| 19 | 19_coffee_automaton.ipynb | pt_19_coffee_automaton.ipynb | todo |
| 20 | 20_neural_turing_machine.ipynb | pt_20_neural_turing_machine.ipynb | todo |
| 21 | 21_ctc_speech.ipynb | pt_21_ctc_speech.ipynb | todo |
| 22 | 22_scaling_laws.ipynb | pt_22_scaling_laws.ipynb | todo |
| 23 | 23_mdl_principle.ipynb | pt_23_mdl_principle.ipynb | todo |
| 24 | 24_machine_super_intelligence.ipynb | pt_24_machine_super_intelligence.ipynb | todo |
| 25 | 25_kolmogorov_complexity.ipynb | pt_25_kolmogorov_complexity.ipynb | todo |
| 26 | 26_cs231n_cnn_fundamentals.ipynb | pt_26_cs231n_cnn_fundamentals.ipynb | todo |
| 27 | 27_multi_token_prediction.ipynb | pt_27_multi_token_prediction.ipynb | todo |
| 28 | 28_dense_passage_retrieval.ipynb | pt_28_dense_passage_retrieval.ipynb | todo |
| 29 | 29_rag.ipynb | pt_29_rag.ipynb | todo |
| 30 | 30_lost_in_middle.ipynb | pt_30_lost_in_middle.ipynb | todo |

## Suggested First Notebook
Start with `pt_02_char_rnn_karpathy.ipynb` because it is simple, visual, and teaches core training loop concepts.

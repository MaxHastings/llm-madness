# llm-madness

A lightweight, end-to-end text LLM pipeline with a web UI for configs, datasets, and run inspection.

<img width="1728" height="993" alt="Screenshot 2026-01-01 at 9 07 46 PM" src="https://github.com/user-attachments/assets/8518f0e6-a1a1-42ad-a3a9-2031e1103327" />

<img width="1728" height="992" alt="Screenshot 2026-01-01 at 9 07 16 PM" src="https://github.com/user-attachments/assets/eea08ced-4e53-409e-bfab-5e17b33d2922" />

<img width="1728" height="995" alt="Screenshot 2026-01-01 at 9 09 05 PM" src="https://github.com/user-attachments/assets/3fb88dce-4b4f-47e1-99ec-47fc692345cb" />

## What it does

- Tokenizer: configure BPE settings and build vocabularies from dataset snapshots.
- Training: configure model + optimizer settings and launch runs.
- Datasets: combine multiple `data/**.txt` inputs into a manifest and materialized snapshot.
- Artifacts: every run writes a `run.json` plus stage-specific outputs under `runs/`.
- Inspect: view loss curves, samples, tokenizer stats, and per-token top-k to debug behavior.

## Core concepts

### **1. Custom LLM Training Pipeline**
- **GPT-style transformer implementation** (`llm_madness/model.py`)
  - Causal self-attention with multi-head architecture
  - Configurable model dimensions (layers, heads, embedding size, block size)
  - Built-in generation with temperature and top-k sampling
  - Forward passes with attention trace and hidden state inspection
  
- **Training infrastructure** (`llm_madness/stages/train.py`)
  - AdamW optimizer with learning rate warmup and cosine decay
  - Automatic device selection (CUDA/MPS/CPU)
  - Token cache to avoid re-tokenizing large snapshots each run (`training.token_cache`)
  - Gradient clipping and dropout regularization
  - Checkpoint saving at intervals
  - Live sampling during training to monitor behavior

### **2. Tokenizer Experimentation**
- **BPE tokenizer training** (`llm_madness/tokenizer.py`)
  - Byte-level tokenization with customizable vocabulary sizes
  - Special token discovery via regex patterns
  - Configurable pre-tokenization (digit splitting, prefix spaces)
  - Vocabulary versioning and manifest tracking

### **3. Dataset Management**
- **Dataset manifests** (`llm_madness/datasets/manifest.py`)
  - Combine multiple text files into versioned snapshots
  - SHA-256 hashing for reproducibility
  - Automatic train/validation splitting
  - Track dataset lineage and sources
- **Token datasets** (`llm_madness/stages/tokenize_dataset.py`)
  - Pre-tokenize snapshots into `runs/tokens/**/train.bin` + `val.bin` for fast training startup
  - Memory-map token binaries during training to avoid holding full corpora in RAM

### **4. Web-Based Inspector UI**
A **built-in Python web interface** for:
- **Tokenizer Config Management**: Create, version, and compare tokenizer configurations
- **Training Config Management**: Design model architectures and training hyperparameters
- **Dataset Browser**: View and organize training datasets
- **Run Inspection**: 
  - Real-time loss curve visualization
  - Sample generation monitoring
  - Per-layer attention inspection
  - Token-level top-k prediction analysis
  - Training log streaming

### **5. Experiment Tracking & Reproducibility**
- **Manifest system** (`runs/`)
  - Every run generates a `run.json` with inputs, outputs, config, and git SHA
  - Automatic artifact organization by stage (tokenizer/train/pipeline)
  - Complete provenance tracking from data → tokenizer → model

### **6. Pipeline Orchestration**
- **End-to-end pipeline** (`llm_madness/stages/pipeline.py`)
  - Chain tokenizer training → model training
  - Automatic resolution of "latest" runs
  - Configurable stage enabling/disabling
  - Error handling with status tracking

---

## **Practical Use Cases for AI Engineers**

### **Research & Experimentation**
- **Architecture search**:  Quickly iterate on model dimensions (layers, heads, embedding sizes)
- **Tokenizer optimization**: Test different vocabulary sizes and special token strategies
- **Training dynamics**: Study loss curves, convergence patterns, and learning rate schedules
- **Ablation studies**: Version configs to compare different hyperparameters

### **Educational & Learning**
- **Understand transformer internals**: Clean, readable implementation of GPT architecture
- **Debug attention patterns**: Visualize attention maps per layer and head
- **Token prediction analysis**: See what the model "thinks" at each position

### **Custom Domain Applications**
- **Domain-specific tokenization**: Train BPE on specialized corpora (code, math, medical texts)
- **Small model deployment**: Train tiny models for edge devices or specific tasks
- **Data exploration**: Use the dataset tools to analyze text statistics

### **Prototyping & Validation**
- **Proof-of-concept models**:  Quickly train small models to validate ideas
- **Synthetic data experiments**: Test with custom-generated datasets
- **Baseline establishment**: Create reference models for comparison

---

## **Key Technical Features**

### **Model Architecture** (`model.py`)
```python
- CausalSelfAttention with multi-head attention
- MLP feed-forward blocks (4x expansion)
- Layer normalization (pre-norm architecture)
- Position + token embeddings
- Temperature + top-k sampling generation
- Trace mode for debugging (attention maps, MLP outputs, hidden states)
```

### **Training Loop** (`stages/train.py`)
```python
- Learning rate warmup + cosine annealing
- Train/validation splitting with per-split loss estimation
- Automatic perplexity calculation
- Live sample generation at eval intervals
- JSONL logging for loss/perplexity/samples
- Checkpoint management with "latest" symlink
```

### **Configuration System**
- JSON-based configs with dot-path overrides (`--set model.n_layer=8`)
- Metadata tracking (name, version, parent_id, timestamps)
- Default config templates in `configs/`

### **Model Flags (Optional)**
- `use_rmsnorm`: swap LayerNorm → RMSNorm
- `use_swiglu`: gated FFN (SwiGLU) instead of GELU MLP
- `use_rope`: rotary position embeddings (disables absolute `pos_emb`)
- `use_sdpa`: use PyTorch scaled dot-product attention on CUDA (CPU/MPS fallback to manual path)
- `use_kv_cache`: cache K/V during generation for faster autoregressive decode
- Example: `configs/training/modern_tiny__v001.json` enables a modern tiny stack



## **Getting Started as an AI Engineer**

### **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Run an end-to-end pipeline (data → tokenizer → train)
python -m scripts.pipeline --config configs/pipeline.json

# Launch web UI (inspect the latest training run)
python -m scripts.web_ui --run-dir runs/train/latest

# Or use CLI directly
python -m scripts.train_tokenizer --config configs/tokenizer/default__v002.json
python -m scripts.tokenize_dataset --config configs/tokenize_dataset/default__v001.json --dataset-manifest runs/datasets/latest/dataset_manifest.json
python -m scripts.train_model --config configs/training/default__v001.json
python -m scripts.pipeline --config configs/pipeline.json
```

### **Example Workflow**
1. Add training data to `data/*.txt`
2. Create dataset snapshot via Web UI or CLI
3. Train tokenizer on snapshot
4. Configure model architecture
5. Train model with real-time monitoring
6. Inspect results:  loss curves, samples, attention patterns

---

## **Best For**
- Rapid prototyping of small LLMs  
- Educational purposes (understanding transformers)  
- Domain-specific tokenizer research  
- Debugging model behavior with transparency  
- Local experimentation without heavy infrastructure

## **Non-goals**
- Not a hosted experiment tracker (no accounts, remote backend, or team features)
- Not designed for large-scale or distributed training
- Not a general-purpose ML framework (focused on tiny, reproducible text GPT experiments)

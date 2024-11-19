# CLEAR-CausaL-Enhanced-Avatar-Retrieval

This repository contains the implementation of CLEAR, a causal learning enhanced version of the AVATAR model for retrieval tasks.

## Features

- Enhanced AVATAR model with causal learning capabilities
- Causal graph learning for better feature representation
- DAG constraint for ensuring valid causal relationships
- Compatible with original AVATAR training pipeline

## Installation

```bash
git clone https://github.com/SnSnLi/CLEAR-CausaL-Enhanced-Avatar-Retrieval.git
cd CLEAR-CausaL-Enhanced-Avatar-Retrieval
pip install -r requirements.txt

## Train
To train the model with causal learning:
bash scripts/run_avatar_flickr30k_entities.sh

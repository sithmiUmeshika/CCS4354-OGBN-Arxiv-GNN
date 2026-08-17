# CCS4354 – OGBN-Arxiv Graph Neural Network Coursework

## Project Overview
This project implements a complete Graph Neural Network workflow for node classification on the OGBN-Arxiv citation network.

Two GNN architectures were implemented and compared:

- Graph Convolutional Network (GCN)
- GraphSAGE

## Dataset
- Nodes: 169,343
- Directed Edges: 1,166,243
- Node Features: 128
- Classes: 40
- Task: Multi-class node classification

## Main Components
- Tensor fundamentals
- Graph representation and analysis
- Data preprocessing and normalization
- GCN model development
- GraphSAGE model development
- Training and hyperparameter optimization
- Model evaluation
- PCA embedding visualization
- Neighborhood influence analysis
- Edge ablation analysis
- Streamlit Graph Intelligence Dashboard

## Technologies
- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- Scikit-learn
- Pandas
- Matplotlib
- Streamlit

## Project Structure
- `01_Source_Code` – Python source code
- `02_Jupyter_Notebook` – Final Jupyter Notebook
- `03_Trained_Models` – Saved GCN and GraphSAGE models
- `04_Technical_Report` – Technical report PDF
- `05_Streamlit_Dashboard` – Streamlit dashboard files
- `06_Presentation` – Final presentation slides

## Models
### GCN
Graph Convolutional Network using graph convolution to combine node features with citation-neighbourhood information.

### GraphSAGE
GraphSAGE using mean neighbourhood aggregation to learn node representations.

## Explainability
Three explainability approaches were included:
- PCA node embedding visualization
- Neighborhood influence analysis
- Edge ablation analysis

## Dashboard
The Streamlit dashboard displays:
- Graph statistics
- GCN and GraphSAGE performance
- Node-level classification results
- PCA node embeddings
- Sample citation subgraph
- Edge-ablation visualization

## Module
**Module Code:** CCS4354  
**Module Name:** Tensors and Graphs  
**Dataset:** OGBN-Arxiv

## Group Members
1. sithmi umeshika – 23UG1-0078
2. Samoshi rupasingha – 23UG1-0118
3. Gihan chathuranga – 23UG1-0119
4. Dewmi Abeykoon – CIT-23-02-0090

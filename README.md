# CCS4354 – OGBN-Arxiv Graph Neural Network Coursework
Group members- 23UG1-0078
               23UG1-0118
               23UG1-0119
               CIT-23-02-0090

## Project Overview

This project implements a complete Graph Neural Network workflow for node classification on the OGBN-Arxiv citation network.

Two GNN architectures were developed and compared:

- Graph Convolutional Network (GCN)
- GraphSAGE

The project includes tensor operations, graph analysis, preprocessing, model development, training, hyperparameter optimization, evaluation, explainability analysis, and a Streamlit dashboard.

## Dataset

OGBN-Arxiv

- Nodes: 169,343
- Directed Edges: 1,166,243
- Node Features: 128
- Classes: 40
- Task: Multi-class node classification

## Models

### Graph Convolutional Network
GCN uses graph convolution to combine node features with information from neighbouring citation nodes.

### GraphSAGE
GraphSAGE uses mean neighbourhood aggregation to learn representations from the target node and its neighbouring papers.

## Project Structure

```text
notebook/
    Final Jupyter notebook

models/
    Trained GCN and GraphSAGE models
    Model configuration

dashboard/
    Streamlit application and exported result files

report/
    Technical report

presentation/
    Presentation slides

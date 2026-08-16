
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=(
        "OGBN-Arxiv Graph Intelligence Dashboard"
    ),
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "OGBN-Arxiv Graph Intelligence Dashboard"
)

st.caption(
    "CCS4354 – Tensors and Graphs Coursework"
)

st.markdown(
    """
    This dashboard presents graph statistics,
    GCN and GraphSAGE performance, node-level
    classification results, and learned node
    embedding visualizations for the OGBN-Arxiv
    citation network.
    """
)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_graph_stats():

    with open(
        BASE_DIR / "graph_stats.json",
        "r"
    ) as file:

        return json.load(
            file
        )


@st.cache_data
def load_model_results():

    return pd.read_csv(
        BASE_DIR
        / "model_evaluation_results.csv",
        index_col=0
    )


@st.cache_data
def load_predictions():

    return pd.read_csv(
        BASE_DIR
        / "predictions.csv"
    )


# ============================================================
# LOAD DATA
# ============================================================

stats = load_graph_stats()

model_results = (
    load_model_results()
)

predictions = (
    load_predictions()
)


# ============================================================
# 1. GRAPH STATISTICS
# ============================================================

st.header(
    "1. Graph Statistics"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Nodes",
        f"{stats['nodes']:,}"
    )

with col2:

    st.metric(
        "Directed Edges",
        f"{stats['directed_edges']:,}"
    )

with col3:

    st.metric(
        "Node Features",
        stats["features"]
    )

with col4:

    st.metric(
        "Classes",
        stats["classes"]
    )


col5, col6, col7, col8 = (
    st.columns(4)
)

with col5:

    st.metric(
        "Message-Passing Edges",
        f"{stats['message_passing_edges']:,}"
    )

with col6:

    st.metric(
        "Training Nodes",
        f"{stats['training_nodes']:,}"
    )

with col7:

    st.metric(
        "Validation Nodes",
        f"{stats['validation_nodes']:,}"
    )

with col8:

    st.metric(
        "Test Nodes",
        f"{stats['test_nodes']:,}"
    )


st.metric(
    "Directed Graph Density",
    f"{stats['density']:.8f}"
)

st.caption(
    "The low graph density demonstrates that "
    "OGBN-Arxiv is a highly sparse citation network."
)


# ============================================================
# 2. MODEL PERFORMANCE
# ============================================================

st.divider()

st.header(
    "2. Model Performance"
)

st.markdown(
    """
    The following table compares GCN and GraphSAGE
    on the validation and test subsets using accuracy,
    precision, recall, weighted F1 and macro F1 metrics.
    """
)

st.dataframe(
    model_results.round(4),
    width="stretch"
)


# ============================================================
# PERFORMANCE CHART
# ============================================================

available_chart_columns = [
    column
    for column in [
        "Accuracy",
        "F1_Weighted",
        "F1_Macro"
    ]
    if column
    in model_results.columns
]

if available_chart_columns:

    st.subheader(
        "Performance Comparison"
    )

    st.bar_chart(
        model_results[
            available_chart_columns
        ]
    )


# ============================================================
# 3. NODE CLASSIFICATION RESULTS
# ============================================================

st.divider()

st.header(
    "3. Node Classification Results"
)

st.markdown(
    """
    Select a node to inspect its true research category,
    GCN prediction, GraphSAGE prediction, dataset split,
    and whether each prediction was correct.
    """
)

node_id = st.number_input(
    "Enter Node ID",
    min_value=0,
    max_value=(
        len(predictions) - 1
    ),
    value=0,
    step=1
)

selected_row = predictions[
    predictions["Node_ID"]
    == node_id
]

if not selected_row.empty:

    row = selected_row.iloc[0]

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "True Class",
            int(
                row[
                    "True_Label"
                ]
            )
        )

    with col2:

        st.metric(
            "GCN Prediction",
            int(
                row[
                    "GCN_Prediction"
                ]
            )
        )

    with col3:

        st.metric(
            "GraphSAGE Prediction",
            int(
                row[
                    "GraphSAGE_Prediction"
                ]
            )
        )

    with col4:

        st.metric(
            "Dataset Split",
            row["Split"]
        )

    st.write(
        "GCN Correct:",
        bool(
            row[
                "GCN_Correct"
            ]
        )
    )

    st.write(
        "GraphSAGE Correct:",
        bool(
            row[
                "GraphSAGE_Correct"
            ]
        )
    )

    st.dataframe(
        selected_row,
        width="stretch"
    )

else:

    st.warning(
        "The selected node was not found."
    )


# ============================================================
# NODE RESULT FILTER
# ============================================================

st.subheader(
    "Browse Node Predictions"
)

selected_split = st.selectbox(
    "Filter by Dataset Split",
    [
        "All",
        "Training",
        "Validation",
        "Test"
    ]
)

if selected_split == "All":

    filtered_predictions = (
        predictions
    )

else:

    filtered_predictions = (
        predictions[
            predictions[
                "Split"
            ]
            == selected_split
        ]
    )

st.dataframe(
    filtered_predictions.head(100),
    width="stretch"
)

st.caption(
    "The first 100 matching node records are shown."
)


# ============================================================
# 4. PCA NODE EMBEDDING VISUALIZATION
# ============================================================

st.divider()

st.header(
    "4. Learned Node Embeddings"
)

st.markdown(
    """
    PCA reduces the learned GCN hidden node
    representations to two dimensions so that
    the structure of the learned embedding space
    can be visualized.
    """
)

pca_path = (
    BASE_DIR
    / "pca_embeddings.png"
)

if pca_path.exists():

    st.image(
        str(pca_path),
        caption=(
            "PCA visualization of learned "
            "GCN node embeddings"
        ),
        width="stretch"
    )

else:

    st.warning(
        "PCA embedding image is not available."
    )


# ============================================================
# 5. SAMPLE CITATION SUBGRAPH
# ============================================================

st.divider()

st.header(
    "5. Sample Citation Subgraph"
)

graph_path = (
    BASE_DIR
    / "sample_subgraph.png"
)

if graph_path.exists():

    st.image(
        str(graph_path),
        caption=(
            "Sample of the directed "
            "OGBN-Arxiv citation network"
        ),
        width="stretch"
    )

else:

    st.warning(
        "Sample subgraph image is not available."
    )


# ============================================================
# 6. OPTIONAL EDGE ABLATION VISUALIZATION
# ============================================================

edge_ablation_path = (
    BASE_DIR
    / "edge_ablation_confidence.png"
)

if edge_ablation_path.exists():

    st.divider()

    st.header(
        "6. Additional Explainability"
    )

    st.image(
        str(edge_ablation_path),
        caption=(
            "Effect of neighborhood edge removal "
            "on GCN prediction confidence"
        ),
        width="stretch"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Graph Intelligence Dashboard developed for "
    "CCS4354 – Tensors and Graphs using the "
    "OGBN-Arxiv citation network."
)

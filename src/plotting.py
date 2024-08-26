import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv
import seaborn as sns
import math


def plot_time_series(data, columns, title="Time Series Plot"):
    plt.figure(figsize=(12, 8))
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label = columns[i])
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    return plt.gcf()


def plot_heatmap(adjacency_matrices, columns, title="Heatmap of Adjacency Matrices"):
    num_matrices = len(adjacency_matrices)
    
    # set figure layout
    if num_matrices <= 3:
        nrows, ncols = 1, num_matrices
    else:
        nrows = math.ceil(num_matrices / 3)
        ncols = min(num_matrices, 3)
    
    # set figure size
    fig_width = 5 * ncols
    fig_height = 4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    
    # If there is only one subplot, convert axes to a 2D array for uniform processing
    if num_matrices == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    
    # set y labels
    y_labels = [f"{col}(t)" for col in columns]
    
    for i, matrix in enumerate(adjacency_matrices):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        # set x labels
        if i == 0:
            x_labels = y_labels
        else:
            x_labels = [f"{col}(t-{i})" for col in columns]
        
        sns.heatmap(matrix, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f', 
                    xticklabels=x_labels, yticklabels=y_labels)
        ax.set_title(f"Heatmap with Lag {i}")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # remove extra subplots
    for i in range(num_matrices, nrows * ncols):
        row = i // ncols
        col = i % ncols
        fig.delaxes(axes[row, col])
    
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    return plt.gcf()


def plot_causal_graph(
    adjacency_matrices,
    node_labels=None,
    title="Causal Graph",
    lower_limit=0.01,
    ignore_shape=False
):
    num_layers = len(adjacency_matrices)
    num_nodes_per_layer = adjacency_matrices[0].shape[0]
    
    # Automatically generate full labels
    labels = []
    for t in range(num_layers):
        for node in node_labels:
            if t == 0:
                labels.append(f"{node}(t)")
            else:
                labels.append(f"{node}(t-{t})")
    
    d = gv.Digraph(engine="fdp")
    d.attr(rankdir="LR", splines="curved")
    d.attr(label=title, labelloc="t", labeljust="c")
    
    # Create nodes with specific positions
    for layer in range(num_layers):
        for node in range(num_nodes_per_layer):
            node_name = labels[layer * num_nodes_per_layer + node] if labels else f"x{node}(t-{num_layers-layer-1})"
            x_pos = (num_layers - layer - 1) * 5
            y_pos = (num_nodes_per_layer - node - 1) * 2
            d.node(node_name, pos=f"{x_pos},{y_pos}!")
    
    base_colors = ["red", "green", "blue"]  # Define colors for layers
    
    # Add edges
    for layer in range(num_layers):
        B = adjacency_matrices[layer]
        idx = np.abs(B) > lower_limit
        dirs = np.where(idx)
        color = base_colors[layer % len(base_colors)]
        for to, from_, coef in zip(dirs[0], dirs[1], B[idx]):
            from_name = labels[layer * num_nodes_per_layer + from_] if labels else f"x{from_}(t-{num_layers-layer-1})"
            to_name = labels[to] if layer > 0 else labels[to]
            d.edge(from_name, to_name, 
                   label=f"<<b>{coef:.2f}</b>>",  # Bold label with coefficient value
                   color=color,  # Edge color based on layer
                   fontsize="10",
                   decorate="true",  # Makes the label follow the line
                   labelloc="t",  # Places the label above the line
                   penwidth="1",
                   arrowsize="0.5")

    return d

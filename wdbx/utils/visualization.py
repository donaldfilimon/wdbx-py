"""
Visualization utilities for WDBX.

This module provides visualization tools for vector data.
"""

import logging
import io
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VectorVisualizer:
    """
    Vector visualization tools.

    This class provides methods for visualizing vector data in various formats.

    Attributes:
        wdbx: Reference to the WDBX instance
    """

    def __init__(self, wdbx=None):
        """
        Initialize the visualizer.

        Args:
            wdbx: Optional reference to the WDBX instance
        """
        self.wdbx = wdbx

    def reduce_dimensions(
        self,
        vectors: List[List[float]],
        method: str = "pca",
        n_components: int = 2,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Reduce dimensionality of vectors for visualization.

        Args:
            vectors: List of vectors to reduce
            method: Dimensionality reduction method ('pca', 't-sne', or 'umap')
            n_components: Number of components in output
            random_state: Random state for reproducibility

        Returns:
            Reduced vectors as numpy array

        Raises:
            ValueError: If the method is not supported
            ImportError: If required dependencies are not installed
        """
        # Convert to numpy array
        vectors_np = np.array(vectors)

        if method == "pca":
            try:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=n_components, random_state=random_state)
                return pca.fit_transform(vectors_np)
            except ImportError:
                logger.error("scikit-learn not installed, required for PCA")
                raise ImportError(
                    "scikit-learn is required for PCA. Install with: pip install scikit-learn"
                )

        elif method == "t-sne":
            try:
                from sklearn.manifold import TSNE

                tsne = TSNE(n_components=n_components, random_state=random_state)
                return tsne.fit_transform(vectors_np)
            except ImportError:
                logger.error("scikit-learn not installed, required for t-SNE")
                raise ImportError(
                    "scikit-learn is required for t-SNE. Install with: pip install scikit-learn"
                )

        elif method == "umap":
            try:
                import umap

                reducer = umap.UMAP(
                    n_components=n_components, random_state=random_state
                )
                return reducer.fit_transform(vectors_np)
            except ImportError:
                logger.error("umap-learn not installed, required for UMAP")
                raise ImportError(
                    "umap-learn is required for UMAP. Install with: pip install umap-learn"
                )

        else:
            raise ValueError(f"Unsupported dimension reduction method: {method}")

    def plot_vectors(
        self,
        vectors: List[List[float]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        method: str = "pca",
        n_components: int = 2,
        title: str = "Vector Space Visualization",
        width: int = 800,
        height: int = 600,
        return_type: str = "html",
    ) -> str:
        """
        Create a plot of vector data.

        Args:
            vectors: List of vectors to plot
            labels: Optional list of labels for each vector
            colors: Optional list of colors for each vector
            method: Dimensionality reduction method ('pca', 't-sne', or 'umap')
            n_components: Number of components to reduce to (2 or 3)
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            return_type: Return type ('html', 'json', or 'base64')

        Returns:
            Visualization in the specified format

        Raises:
            ValueError: If parameters are invalid
            ImportError: If required dependencies are not installed
        """
        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3")

        # Reduce dimensions
        reduced_vectors = self.reduce_dimensions(
            vectors, method, n_components, random_state=42
        )

        # Default labels and colors
        if labels is None:
            labels = [f"Vector {i+1}" for i in range(len(vectors))]

        if colors is None:
            # Generate colors based on labels
            unique_labels = list(set(labels))
            label_to_color = {}

            # Use qualitative colormap
            try:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors

                colormap = plt.cm.get_cmap("tab10", len(unique_labels))
                for i, label in enumerate(unique_labels):
                    label_to_color[label] = mcolors.rgb2hex(colormap(i))
            except ImportError:
                # Fallback to basic colors
                basic_colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ]
                for i, label in enumerate(unique_labels):
                    label_to_color[label] = basic_colors[i % len(basic_colors)]

            colors = [label_to_color[label] for label in labels]

        # Create plot
        if n_components == 2:
            return self._plot_2d(
                reduced_vectors, labels, colors, title, width, height, return_type
            )
        else:
            return self._plot_3d(
                reduced_vectors, labels, colors, title, width, height, return_type
            )

    def _plot_2d(
        self,
        vectors: np.ndarray,
        labels: List[str],
        colors: List[str],
        title: str,
        width: int,
        height: int,
        return_type: str,
    ) -> str:
        """
        Create a 2D plot.

        Args:
            vectors: Reduced vectors (n_samples, 2)
            labels: Labels for each vector
            colors: Colors for each vector
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            return_type: Return type ('html', 'json', or 'base64')

        Returns:
            Visualization in the specified format
        """
        try:
            import plotly.graph_objects as go
            from plotly.utils import PlotlyJSONEncoder
            import json

            # Create figure
            fig = go.Figure()

            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=vectors[:, 0],
                    y=vectors[:, 1],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=colors,
                        opacity=0.8,
                        line=dict(width=1, color="white"),
                    ),
                    text=labels,
                    hoverinfo="text",
                )
            )

            # Update layout
            fig.update_layout(
                title=title,
                autosize=False,
                width=width,
                height=height,
                plot_bgcolor="rgba(240, 240, 240, 0.8)",
                hovermode="closest",
                xaxis=dict(
                    title="Component 1",
                    gridcolor="white",
                    gridwidth=1,
                    zerolinecolor="white",
                    zerolinewidth=2,
                ),
                yaxis=dict(
                    title="Component 2",
                    gridcolor="white",
                    gridwidth=1,
                    zerolinecolor="white",
                    zerolinewidth=2,
                ),
                margin=dict(l=40, r=40, b=40, t=60),
            )

            # Return in requested format
            if return_type == "html":
                return fig.to_html(include_plotlyjs="cdn", full_html=False)
            elif return_type == "json":
                return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
            elif return_type == "base64":
                buffer = io.BytesIO()
                fig.write_image(buffer, format="png")
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8")
            else:
                raise ValueError(f"Unsupported return_type: {return_type}")

        except ImportError:
            logger.error("plotly not installed, required for visualization")
            raise ImportError(
                "plotly is required for visualization. Install with: pip install plotly kaleido"
            )

    def _plot_3d(
        self,
        vectors: np.ndarray,
        labels: List[str],
        colors: List[str],
        title: str,
        width: int,
        height: int,
        return_type: str,
    ) -> str:
        """
        Create a 3D plot.

        Args:
            vectors: Reduced vectors (n_samples, 3)
            labels: Labels for each vector
            colors: Colors for each vector
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            return_type: Return type ('html', 'json', or 'base64')

        Returns:
            Visualization in the specified format
        """
        try:
            import plotly.graph_objects as go
            from plotly.utils import PlotlyJSONEncoder
            import json

            # Create figure
            fig = go.Figure()

            # Add scatter plot
            fig.add_trace(
                go.Scatter3d(
                    x=vectors[:, 0],
                    y=vectors[:, 1],
                    z=vectors[:, 2],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=colors,
                        opacity=0.8,
                        line=dict(width=0.5, color="white"),
                    ),
                    text=labels,
                    hoverinfo="text",
                )
            )

            # Update layout
            fig.update_layout(
                title=title,
                autosize=False,
                width=width,
                height=height,
                scene=dict(
                    xaxis=dict(title="Component 1"),
                    yaxis=dict(title="Component 2"),
                    zaxis=dict(title="Component 3"),
                ),
                margin=dict(l=0, r=0, b=0, t=50),
            )

            # Return in requested format
            if return_type == "html":
                return fig.to_html(include_plotlyjs="cdn", full_html=False)
            elif return_type == "json":
                return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
            elif return_type == "base64":
                buffer = io.BytesIO()
                fig.write_image(buffer, format="png")
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8")
            else:
                raise ValueError(f"Unsupported return_type: {return_type}")

        except ImportError:
            logger.error("plotly not installed, required for visualization")
            raise ImportError(
                "plotly is required for visualization. Install with: pip install plotly kaleido"
            )

    def plot_similarity_matrix(
        self,
        vectors: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "Vector Similarity Matrix",
        width: int = 800,
        height: int = 800,
        return_type: str = "html",
    ) -> str:
        """
        Create a similarity matrix heatmap.

        Args:
            vectors: List of vectors to compare
            labels: Optional list of labels for each vector
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            return_type: Return type ('html', 'json', or 'base64')

        Returns:
            Visualization in the specified format

        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            import plotly.graph_objects as go
            from plotly.utils import PlotlyJSONEncoder
            import json

            # Convert to numpy array
            vectors_np = np.array(vectors)

            # Normalize vectors
            norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
            normalized_vectors = vectors_np / norms

            # Compute similarity matrix (cosine similarity)
            similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)

            # Default labels
            if labels is None:
                labels = [f"Vector {i+1}" for i in range(len(vectors))]

            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=similarity_matrix,
                    x=labels,
                    y=labels,
                    colorscale="Viridis",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title="Similarity"),
                )
            )

            # Update layout
            fig.update_layout(
                title=title,
                autosize=False,
                width=width,
                height=height,
                margin=dict(l=0, r=0, b=0, t=50),
            )

            # Return in requested format
            if return_type == "html":
                return fig.to_html(include_plotlyjs="cdn", full_html=False)
            elif return_type == "json":
                return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
            elif return_type == "base64":
                buffer = io.BytesIO()
                fig.write_image(buffer, format="png")
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8")
            else:
                raise ValueError(f"Unsupported return_type: {return_type}")

        except ImportError:
            logger.error("plotly not installed, required for visualization")
            raise ImportError(
                "plotly is required for visualization. Install with: pip install plotly kaleido"
            )

    def visualize_vectors_from_db(
        self,
        wdbx=None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        max_vectors: int = 1000,
        method: str = "pca",
        n_components: int = 2,
        title: str = "Vector Database Visualization",
        return_type: str = "html",
    ) -> str:
        """
        Visualize vectors from the database.

        Args:
            wdbx: WDBX instance (uses self.wdbx if None)
            filter_metadata: Optional metadata filter
            max_vectors: Maximum number of vectors to visualize
            method: Dimensionality reduction method
            n_components: Number of components to reduce to
            title: Plot title
            return_type: Return type

        Returns:
            Visualization in the specified format

        Raises:
            ValueError: If WDBX instance is not available
        """
        # Use provided WDBX instance or default
        wdbx = wdbx or self.wdbx

        if wdbx is None:
            raise ValueError("WDBX instance is required")

        # Get all vectors
        vectors = []
        labels = []
        colors = []

        # Search with optional filter
        results = wdbx.vector_search(
            query_vector=[0.1] * wdbx.vector_dim,  # Dummy query vector
            limit=max_vectors,
            threshold=0.0,
            filter_metadata=filter_metadata,
        )

        # Extract vectors and metadata
        for vector_id, similarity, metadata in results:
            vector_data = wdbx.get_vector(vector_id)
            if vector_data:
                vector, _ = vector_data
                vectors.append(vector)

                # Extract label and color from metadata
                label = metadata.get("label", vector_id)
                labels.append(str(label))

                # Use category or source for coloring
                category = metadata.get("category", metadata.get("source", "default"))
                colors.append(str(category))

        # Plot vectors
        return self.plot_vectors(
            vectors=vectors,
            labels=labels,
            method=method,
            n_components=n_components,
            title=title,
            return_type=return_type,
        )

    def create_interactive_dashboard(
        self, wdbx=None, port: int = 8050, debug: bool = False
    ) -> None:
        """
        Create an interactive dashboard for vector visualization.

        Args:
            wdbx: WDBX instance (uses self.wdbx if None)
            port: Port to run the dashboard on
            debug: Whether to run in debug mode

        Raises:
            ValueError: If WDBX instance is not available
            ImportError: If required dependencies are not installed
        """
        # Use provided WDBX instance or default
        wdbx = wdbx or self.wdbx

        if wdbx is None:
            raise ValueError("WDBX instance is required")

        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output, State
            import dash_bootstrap_components as dbc
        except ImportError:
            logger.error(
                "dash and dash-bootstrap-components not installed, required for dashboard"
            )
            raise ImportError(
                "dash and dash-bootstrap-components are required for the dashboard. "
                "Install with: pip install dash dash-bootstrap-components"
            )

        # Create Dash app
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="WDBX Vector Visualization",
        )

        # Define layout
        app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1("WDBX Vector Visualization Dashboard"),
                                html.Hr(),
                            ],
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Visualization Settings"),
                                        dbc.CardBody(
                                            [
                                                dbc.Form(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Label(
                                                                            "Dimension Reduction Method:"
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="method-dropdown",
                                                                            options=[
                                                                                {
                                                                                    "label": "PCA",
                                                                                    "value": "pca",
                                                                                },
                                                                                {
                                                                                    "label": "t-SNE",
                                                                                    "value": "t-sne",
                                                                                },
                                                                                {
                                                                                    "label": "UMAP",
                                                                                    "value": "umap",
                                                                                },
                                                                            ],
                                                                            value="pca",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Label(
                                                                            "Number of Components:"
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="components-dropdown",
                                                                            options=[
                                                                                {
                                                                                    "label": "2D",
                                                                                    "value": 2,
                                                                                },
                                                                                {
                                                                                    "label": "3D",
                                                                                    "value": 3,
                                                                                },
                                                                            ],
                                                                            value=2,
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Label(
                                                                            "Max Vectors:"
                                                                        ),
                                                                        dbc.Input(
                                                                            id="max-vectors",
                                                                            type="number",
                                                                            value=500,
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Label(
                                                                            "Filter by Source:"
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="source-dropdown",
                                                                            options=[
                                                                                {
                                                                                    "label": "All",
                                                                                    "value": "all",
                                                                                }
                                                                            ],
                                                                            value="all",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Button(
                                                            "Update Visualization",
                                                            id="update-button",
                                                            color="primary",
                                                            className="mt-3",
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Vector Visualization"),
                                        dbc.CardBody(
                                            [
                                                dcc.Loading(
                                                    id="loading-visualization",
                                                    type="circle",
                                                    children=html.Div(
                                                        id="visualization-content"
                                                    ),
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=8,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Hr(),
                                html.Div(
                                    id="stats-content", className="small text-muted"
                                ),
                            ],
                            width=12,
                        ),
                    ]
                ),
            ],
            fluid=True,
            className="mt-4",
        )

        # Update source dropdown on page load
        @app.callback(
            Output("source-dropdown", "options"), Input("update-button", "n_clicks")
        )
        def update_source_dropdown(n_clicks):
            # Get all unique sources from database
            try:
                results = wdbx.vector_search(
                    query_vector=[0.1] * wdbx.vector_dim,  # Dummy query vector
                    limit=1000,
                    threshold=0.0,
                )

                sources = set()
                for _, _, metadata in results:
                    source = metadata.get("source")
                    if source:
                        sources.add(source)

                options = [{"label": "All", "value": "all"}]
                for source in sorted(sources):
                    options.append({"label": source, "value": source})

                return options
            except Exception as e:
                logger.error(f"Error updating source dropdown: {e}")
                return [{"label": "All", "value": "all"}]

        # Update visualization on button click
        @app.callback(
            Output("visualization-content", "children"),
            Output("stats-content", "children"),
            Input("update-button", "n_clicks"),
            State("method-dropdown", "value"),
            State("components-dropdown", "value"),
            State("max-vectors", "value"),
            State("source-dropdown", "value"),
        )
        def update_visualization(n_clicks, method, n_components, max_vectors, source):
            if n_clicks is None:
                # Initial load
                total_vectors = wdbx.count_vectors()
                return (
                    "Click 'Update Visualization' to generate the visualization.",
                    f"Database contains {total_vectors} vectors.",
                )

            try:
                # Prepare filter
                filter_metadata = None
                if source and source != "all":
                    filter_metadata = {"source": source}

                # Generate visualization
                visualization = self.visualize_vectors_from_db(
                    wdbx=wdbx,
                    filter_metadata=filter_metadata,
                    max_vectors=max_vectors,
                    method=method,
                    n_components=n_components,
                    title=f"Vector Visualization ({method.upper()}, {n_components}D)",
                )

                # Get stats
                total_vectors = wdbx.count_vectors()
                if filter_metadata:
                    stats_text = f"Showing up to {max_vectors} vectors from source '{source}'. Total vectors in database: {total_vectors}."
                else:
                    stats_text = f"Showing up to {max_vectors} vectors from all sources. Total vectors in database: {total_vectors}."

                return (
                    html.Div(dangerouslySetInnerHTML={"__html": visualization}),
                    stats_text,
                )
            except Exception as e:
                logger.error(f"Error updating visualization: {e}")
                return (
                    html.Div(f"Error: {str(e)} "),
                    f"Database contains {
                    wdbx.count_vectors()}  vectors.",
                )

        # Run the app
        app.run_server(host="0.0.0.0", port=port, debug=debug)

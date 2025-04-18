import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from umap import UMAP

class ModelEmbeddingsPlotter:
    def __init__(self, output_dir="models-embeddings"):
        """
        Create or use the given output directory for PNG files.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def _plot_embeddings_and_boundary(self, embeddings, labels, jsonl_name):
        """
        Common routine to project embeddings with UMAP, plot them,
        fit a logistic regression, plot its decision boundary, and
        save the figure as a PNG.
        """
        # Convert labels to np arrays
        labels = np.array(labels)
        num_files = len(labels)

        # 1) UMAP to 2D
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=1.0)
        projs = reducer.fit_transform(embeddings)

        # 2) Plot
        fig, ax = plt.subplots()

        # We'll plot by label M/F
        unique_labels = np.unique(labels)
        for label_value in unique_labels:
            idx = (labels == label_value)
            marker = "x" if label_value == "M" else "o"
            legend_label = "Male speaker" if label_value == "M" else "Female speaker"

            # Plot points
            ax.scatter(
                projs[idx, 0],
                projs[idx, 1],
                marker=marker,
                alpha=0.8,
                label=legend_label
            )

        ax.set_title(f"Embeddings for {num_files} files")
        ax.legend()
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

        # Annotate UMAP parameters
        umap_params = f"UMAP: n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist}"
        fig.text(
            0.5, 0.95, umap_params,
            ha='center',
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="lightgray",
                edgecolor="gray",
                alpha=0.5
            )
        )

        # 3) Fit Logistic Regression for boundary
        numeric_labels = np.where(labels == "M", 0, 1)
        clf = LogisticRegression(solver="lbfgs")
        clf.fit(projs, numeric_labels)

        # 4) Decision Boundary
        x_vals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=200)
        w0, w1 = clf.coef_[0]
        b = clf.intercept_[0]
        y_vals = -(w0 * x_vals + b) / w1

        # Only plot line portion within the visible y-limits
        within_plot = (y_vals > ax.get_ylim()[0]) & (y_vals < ax.get_ylim()[1])
        ax.plot(x_vals[within_plot], y_vals[within_plot], label="Decision boundary", color="red")
        ax.legend()

        # 5) Save the figure
        # e.g. "wav2vec-embeddings.jsonl" -> "wav2vec-embeddings.png"
        out_name = os.path.splitext(os.path.basename(jsonl_name))[0] + ".png"
        out_path = os.path.join(self.output_dir, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved figure] {out_path}")


    def plot_wav2vec_embeddings(self, data_jsonl="wav2vec-embeddings.jsonl"):
        """
        Load Wav2Vec2 embeddings from JSONL, project, plot, and save as PNG.
        """
        with open(data_jsonl, "r") as f_in:
            data = [json.loads(line.strip()) for line in f_in]

        embeddings = np.array([rec["embedding"] for rec in data])
        labels = [rec["label"] for rec in data]

        self._plot_embeddings_and_boundary(embeddings, labels, data_jsonl)

    def plot_resemblyzer_embeddings(self, data_jsonl="resemblyzer-embeddings.jsonl"):
        """
        Load Resemblyzer embeddings from JSONL, project, plot, and save as PNG.
        """
        with open(data_jsonl, "r") as f_in:
            data = [json.loads(line.strip()) for line in f_in]

        embeddings = np.array([rec["embedding"] for rec in data])
        labels = [rec["label"] for rec in data]

        self._plot_embeddings_and_boundary(embeddings, labels, data_jsonl)

    def plot_ecapa_embeddings(self, data_jsonl="ecapa-embeddings.jsonl"):
        """
        Load ECAPA embeddings from JSONL, project, plot, and save as PNG.
        """
        with open(data_jsonl, "r") as f_in:
            data = [json.loads(line.strip()) for line in f_in]

        embeddings = np.array([rec["embedding"] for rec in data])
        labels = [rec["label"] for rec in data]

        self._plot_embeddings_and_boundary(embeddings, labels, data_jsonl)


# ------------------------------
# Example usage (uncomment):
# ------------------------------
if __name__ == "__main__":
    plotter = ModelEmbeddingsPlotter(output_dir="models-embeddings")

    # These will read from:
    #   models-embeddings/wav2vec-embeddings.jsonl
    #   models-embeddings/resemblyzer-embeddings.jsonl
    #   models-embeddings/ecapa-embeddings.jsonl
    # and produce:
    #   models-embeddings/wav2vec-embeddings.png
    #   models-embeddings/resemblyzer-embeddings.png
    #   models-embeddings/ecapa-embeddings.png
    
    plotter.plot_wav2vec_embeddings("models-embeddings/wav2vec-embeddings.jsonl")
    plotter.plot_resemblyzer_embeddings("models-embeddings/resemblyzer-embeddings.jsonl")
    plotter.plot_ecapa_embeddings("models-embeddings/ecapa-embeddings.jsonl")

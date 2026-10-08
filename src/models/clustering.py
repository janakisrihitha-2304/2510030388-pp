from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)


# ============================================================
# LAB 12 - CLUSTERING PLACEMENT DATASET
# ============================================================

print("=" * 60)
print("LAB 12 - CLUSTERING PLACEMENT DATASET")
print("=" * 60)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "src"
    / "data"
    / "raw_placement_data.csv"
)

OUT = (
    ROOT
    / "reports"
    / "figures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

FEATURES = [
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

def load():

    df = pd.read_csv(DATA)

    df.columns = df.columns.str.strip()

    # Convert values such as:
    # Tier 1, Tier 2, Tier 3
    # into numeric values.
    if (
        "college_tier" in df.columns
        and not pd.api.types.is_numeric_dtype(
            df["college_tier"]
        )
    ):

        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
        )

    # Select required features
    df_subset = (
        df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    # Standardize the features
    scaler = StandardScaler()

    X = scaler.fit_transform(df_subset)

    return X


# ------------------------------------------------------------
# CLUSTER EVALUATION
# ------------------------------------------------------------

def scores(X, labels):

    # Ignore DBSCAN noise points
    mask = labels != -1

    X_valid = X[mask]
    labels_valid = labels[mask]

    n_clusters = len(
        set(labels_valid)
    )

    if n_clusters < 2:
        return np.nan, np.nan, n_clusters

    sample_size = min(
        len(X_valid),
        2000
    )

    silhouette = silhouette_score(
        X_valid,
        labels_valid,
        sample_size=sample_size,
        random_state=42
    )

    davies = davies_bouldin_score(
        X_valid,
        labels_valid
    )

    return (
        round(silhouette, 4),
        round(davies, 4),
        n_clusters
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def run():

    print("\nLoading data...")

    X = load()

    n_samples = len(X)

    print(
        f"Dataset loaded with {n_samples} rows."
    )

    # --------------------------------------------------------
    # K-MEANS FOR K = 2 TO 10
    # --------------------------------------------------------

    print(
        "\nRunning K-Means evaluation..."
    )

    sample_size = min(
        n_samples,
        2000
    )

    ks = range(2, 11)

    inertias = []
    silhouettes = []

    for k in ks:

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        labels = model.fit_predict(X)

        inertias.append(
            model.inertia_
        )

        score = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=42
        )

        silhouettes.append(score)

        print(
            f"Finished K={k}"
        )

    # Select K with highest silhouette score
    best_k = list(ks)[
        int(np.argmax(silhouettes))
    ]

    print(
        "\nOptimal K selected by "
        f"highest silhouette: {best_k}"
    )


    # --------------------------------------------------------
    # ELBOW CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(ks),
        inertias,
        marker="o"
    )

    plt.xlabel("K")
    plt.ylabel("Inertia")

    plt.title(
        "Elbow Curve - Placement Dataset"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "clustering_kmeans_elbow.png",
        dpi=150
    )

    plt.close()

    print(
        "Saved: clustering_kmeans_elbow.png"
    )


    # --------------------------------------------------------
    # SILHOUETTE CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(ks),
        silhouettes,
        marker="o"
    )

    plt.xlabel("K")
    plt.ylabel("Silhouette Score")

    plt.title(
        "Silhouette Analysis - Placement Dataset"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "clustering_kmeans_silhouette.png",
        dpi=150
    )

    plt.close()

    print(
        "Saved: clustering_kmeans_silhouette.png"
    )


    # --------------------------------------------------------
    # K-MEANS WITH OPTIMAL K
    # --------------------------------------------------------

    print(
        "\nRunning K-Means with optimal K..."
    )

    kmeans_labels = KMeans(
        n_clusters=best_k,
        n_init=10,
        random_state=42
    ).fit_predict(X)

    km_sil, km_db, km_clusters = scores(
        X,
        kmeans_labels
    )


    # --------------------------------------------------------
    # AGGLOMERATIVE CLUSTERING
    # --------------------------------------------------------

    # Use a sample because the dataset contains
    # 100,000 records.

    rng = np.random.RandomState(42)

    agg_size = min(
        n_samples,
        5000
    )

    agg_indices = rng.choice(
        n_samples,
        agg_size,
        replace=False
    )

    X_agg = X[agg_indices]

    print(
        f"\nRunning Agglomerative Clustering "
        f"on {agg_size} samples..."
    )

    agglomerative_labels = (
        AgglomerativeClustering(
            n_clusters=best_k,
            linkage="ward"
        )
        .fit_predict(X_agg)
    )

    agg_sil, agg_db, agg_clusters = scores(
        X_agg,
        agglomerative_labels
    )


    # --------------------------------------------------------
    # HIERARCHICAL DENDROGRAM
    # --------------------------------------------------------

    print(
        "\nGenerating dendrogram..."
    )

    dendro_size = min(
        n_samples,
        1000
    )

    dendro_indices = rng.choice(
        n_samples,
        dendro_size,
        replace=False
    )

    X_dendro = X[dendro_indices]

    Z = linkage(
        X_dendro,
        method="ward"
    )

    plt.figure(figsize=(12, 6))

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30
    )

    plt.title(
        "Hierarchical Dendrogram - Placement Dataset"
    )

    plt.xlabel(
        "Cluster / Sample Index"
    )

    plt.ylabel("Distance")

    plt.tight_layout()

    plt.savefig(
        OUT / "hierarchical_dendrogram.png",
        dpi=150
    )

    plt.close()

    print(
        "Saved: hierarchical_dendrogram.png"
    )


    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------

    print(
        "\nRunning DBSCAN..."
    )

    dbscan_labels = DBSCAN(
        eps=0.8,
        min_samples=5
    ).fit_predict(X)

    db_sil, db_db, db_clusters = scores(
        X,
        dbscan_labels
    )

    noise_points = int(
        np.sum(dbscan_labels == -1)
    )


    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            [
                "K-Means",
                km_clusters,
                km_sil,
                km_db,
                0
            ],
            [
                "Agglomerative",
                agg_clusters,
                agg_sil,
                agg_db,
                0
            ],
            [
                "DBSCAN",
                db_clusters,
                db_sil,
                db_db,
                noise_points
            ]
        ],
        columns=[
            "Algorithm",
            "Clusters",
            "Silhouette",
            "Davies_Bouldin",
            "Noise_Points"
        ]
    )


    print(
        "\nClustering Comparison:"
    )

    print(
        results.to_string(
            index=False
        )
    )


    # Save comparison
    comparison_file = (
        OUT / "clustering_comparison.csv"
    )

    results.to_csv(
        comparison_file,
        index=False
    )

    print(
        "\nSaved:",
        comparison_file
    )


    print(
        "\nAll outputs and plots saved to:",
        OUT
    )

    print("\n" + "=" * 60)
    print("LAB 12 COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# ------------------------------------------------------------
# START
# ------------------------------------------------------------

if __name__ == "__main__":
    run()
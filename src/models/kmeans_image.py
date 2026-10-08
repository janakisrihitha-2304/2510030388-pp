from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import KMeans


# ============================================================
# LAB 11 - K-MEANS IMAGE SEGMENTATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE = ROOT / "src" / "data" / "input_image.jpg"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


def run():

    print("=" * 60)
    print("LAB 11 - K-MEANS IMAGE SEGMENTATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print("\nLoading image...")

    img = Image.open(IMAGE).convert("RGB").resize((300, 300))

    arr = np.array(img)

    print("Image loaded successfully.")
    print("Image size:", arr.shape)

    # Each pixel becomes one row containing R, G, B
    X = arr.reshape(-1, 3).astype(float)

    print("Number of pixels:", len(X))

    # K values
    ks = [2, 4, 6, 8]

    inertias = []
    segs = []

    # --------------------------------------------------------
    # K-Means for K = 2, 4, 6, 8
    # --------------------------------------------------------

    for k in ks:

        print(f"\nRunning K-Means with K={k}...")

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        labels = model.fit_predict(X)

        # Reconstruct segmented image
        segmented = (
            model.cluster_centers_[labels]
            .reshape(arr.shape)
            .astype("uint8")
        )

        inertias.append(model.inertia_)
        segs.append(segmented)

        # Save individual image
        output_file = OUT / f"segmented_k{k}.png"

        Image.fromarray(segmented).save(output_file)

        print("Saved:", output_file)

    # --------------------------------------------------------
    # Side-by-side comparison
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        1,
        5,
        figsize=(18, 4)
    )

    ax[0].imshow(arr)
    ax[0].set_title("Original")
    ax[0].axis("off")

    for a, k, segmented in zip(
        ax[1:],
        ks,
        segs
    ):

        a.imshow(segmented)
        a.set_title(f"K={k}")
        a.axis("off")

    plt.tight_layout()

    comparison_file = (
        OUT / "image_segmentation_comparison.png"
    )

    plt.savefig(
        comparison_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved:", comparison_file)

    # --------------------------------------------------------
    # Elbow curve
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        ks,
        inertias,
        marker="o",
        linewidth=2
    )

    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve - Image Segmentation")

    plt.grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    plt.tight_layout()

    elbow_file = (
        OUT / "image_segmentation_elbow.png"
    )

    plt.savefig(
        elbow_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", elbow_file)

    print("\n" + "=" * 60)
    print("LAB 11 COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run()
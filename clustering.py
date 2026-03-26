import os
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create output directory
out_dir = "generated_clusters"
os.makedirs(out_dir, exist_ok=True)

summary_stats = []

for d in range(1, 17):
    dataset_id = f"{d:02d}"
    print(f"Processing dataset {dataset_id}...")

    file_path = f"2021-kaust-competition-spatial-statistics-large-datasets/Sub-competition_1a/dataset{dataset_id}_training.csv"
    df = pd.read_csv(file_path)

    coords = df[['x', 'y']].values.astype('float32')
    values = df['values'].values.astype('float32')
    N = len(df)

    # --- kNN ---
    k = 20
    index_flat = faiss.IndexFlatL2(2)
    index_flat.add(coords)
    distances, indices = index_flat.search(coords, k)

    # --- Local mean ---
    local_mean = np.array([
        values[indices[i]].mean()
        for i in range(N)
    ], dtype=np.float32)
    df["local_mean"] = local_mean

    # --- Clustering ---
    features = np.column_stack([df['x'], df['y'], df['local_mean']])
    features = StandardScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=6, random_state=1, n_init=10)
    df["cluster"] = kmeans.fit_predict(features)

    # --- Build graph ---
    rows = np.repeat(np.arange(N), k)
    cols = indices.reshape(-1)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]

    edge_pairs = np.sort(np.vstack([rows, cols]).T, axis=1)
    edges_unique = np.unique(edge_pairs, axis=0)

    adj = [[] for _ in range(N)]
    for u, v in edges_unique:
        adj[u].append(v)
        adj[v].append(u)

    # --- Connected components ---
    cluster_spatial = np.full(N, -1, dtype=int)
    cid = 0

    for cl in np.unique(df["cluster"]):
        nodes = np.where(df["cluster"] == cl)[0]
        visited = set()
        for node in nodes:
            if node in visited:
                continue
            stack = [node]
            comp = []
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                for v in adj[u]:
                    if v in nodes and v not in visited:
                        stack.append(v)
            cluster_spatial[comp] = cid
            cid += 1

    df["cluster_spatial"] = cluster_spatial

    # --- Merge small clusters ---
    n_min = 1000
    counts = pd.Series(cluster_spatial).value_counts()
    small = counts[counts < n_min].index.tolist()
    large = counts[counts >= n_min].index.tolist()

    centroids = df.groupby("cluster_spatial")[["x", "y"]].mean()

    for sc in small:
        idx = np.where(df["cluster_spatial"] == sc)[0]
        neighs = np.unique(indices[idx].reshape(-1))
        neigh_clusters = df.loc[neighs, "cluster_spatial"].unique()
        neigh_clusters = [c for c in neigh_clusters if c in large]

        if len(neigh_clusters) > 0:
            counts_adj = {
                c: np.sum(df["cluster_spatial"].values[indices[idx].reshape(-1)] == c)
                for c in neigh_clusters
            }
            best = max(counts_adj, key=counts_adj.get)
        else:
            c0 = coords[idx].mean(axis=0)
            dists = ((centroids.loc[large][["x", "y"]].values - c0) ** 2).sum(axis=1)
            best = large[np.argmin(dists)]

        df.loc[idx, "cluster_spatial"] = best

    # --- Final connected components ---
    final_labels = np.full(N, -1, dtype=int)
    new_id = 0

    for lab in np.unique(df["cluster_spatial"]):
        idx = np.where(df["cluster_spatial"] == lab)[0]
        visited = set()
        for node in idx:
            if node in visited:
                continue
            stack = [node]
            comp = []
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                for v in adj[u]:
                    if v in idx and v not in visited:
                        stack.append(v)
            final_labels[comp] = new_id
            new_id += 1

    df["cluster_spatial"] = final_labels

    # --- Variance calculations ---
    overall_mean = df["values"].mean()

    within_var = 0.0
    between_var = 0.0

    for cl in df["cluster_spatial"].unique():
        vals = df[df["cluster_spatial"] == cl]["values"]
        mean_cl = vals.mean()
        n_cl = len(vals)

        within_var += ((vals - mean_cl) ** 2).sum()
        between_var += n_cl * (mean_cl - overall_mean) ** 2

    summary_stats.append({
        "dataset": dataset_id,
        "within_variation": within_var,
        "between_variation": between_var
    })

    # --- Save plot ---
    plt.figure(figsize=(8, 8))
    plt.scatter(df["x"], df["y"], c=df["cluster_spatial"], s=0.8, cmap="tab20")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/clusters_{dataset_id}.png", dpi=300)
    plt.close()

    # --- Save dataframe ---
    df.to_csv(f"{out_dir}/dataset{dataset_id}_clusters.csv", index=False)

# --- Save summary ---
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(f"{out_dir}/cluster_variation_summary.csv", index=False)

print("All datasets processed successfully.")
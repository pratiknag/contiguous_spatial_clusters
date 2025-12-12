# !/usr/bin/env Rscript
'''
Created on Friday December 12 

@author: Pratik
'''

library("fields")
df <- read.csv("2021-kaust-competition-spatial-statistics-large-datasets/Sub-competition_1a/dataset01_training.csv", header = T)
quilt.plot(df$x, df$y, df$values, nx = 300, ny = 300)

# install.packages("nabor")



#######################################################################################################
library(nabor)
library(igraph)
library(dplyr)
library(ggplot2)

df <- read.csv("2021-kaust-competition-spatial-statistics-large-datasets/Sub-competition_1a/dataset01_training.csv", header = TRUE)
quilt.plot(df$x, df$y, df$values, nx = 300, ny = 300)

coords <- cbind(df$x, df$y)
k <- 20

nn <- nabor::knn(coords, coords, k)

# local mean 
local_mean <- numeric(nrow(df))
for (i in 1:nrow(df)) {
  local_mean[i] <- mean(df$values[nn$nn.idx[i, ]])
}
df$local_mean <- local_mean

# features + kmeans
features <- scale(cbind(df$x, df$y, df$local_mean))
set.seed(1)
km <- kmeans(features, centers = 6, iter.max = 100)
df$cluster <- km$cluster

# build undirected kNN edge list 
rows <- rep(seq_len(nrow(df)), each = k)
cols <- as.vector(t(nn$nn.idx))
edges <- cbind(rows, cols)
edges <- edges[edges[,1] != edges[,2], ]
# make edges unique & undirected
edges <- unique(t(apply(edges, 1, function(r) sort(r))))
g <- graph_from_edgelist(edges, directed = FALSE)

# split feature clusters into spatially-connected components
df$cluster_spatial <- NA_integer_
cid <- 1L
for (cl in unique(df$cluster)) {
  idx <- which(df$cluster == cl)
  subg <- induced_subgraph(g, idx)
  comps <- components(subg)$membership
  
  for (comp_id in unique(comps)) {
    local_pos <- which(comps == comp_id)
    nodes <- idx[local_pos]
    df$cluster_spatial[nodes] <- cid
    cid <- cid + 1L
  }
}


# Merge small spatial clusters

n_min <- 1000   # change this to desired minimum elements per cluster
tab <- table(df$cluster_spatial)
small_clusters <- as.integer(names(tab[tab < n_min]))
large_clusters <- as.integer(names(tab[tab >= n_min]))

cat("Small clusters:", length(small_clusters), "\n")
cat("Large clusters:", length(large_clusters), "\n")

# Precompute centroids for large clusters
large_centroids <- df %>%
  filter(cluster_spatial %in% large_clusters) %>%
  group_by(cluster_spatial) %>%
  summarise(cx = mean(x), cy = mean(y)) %>%
  ungroup()

# find adjacent large clusters
get_adjacent_large <- function(node_idx, nn_idx_matrix, df, large_ids) {
  # Collect all neighbours of nodes in the small cluster
  neigh_list <- unique(as.vector(nn_idx_matrix[node_idx, , drop = FALSE]))
  neigh_list <- neigh_list[neigh_list %in% seq_len(nrow(df))]
  # map neighbours to their current spatial cluster labels
  neigh_clusters <- unique(df$cluster_spatial[neigh_list])
  # exclude the small cluster itself
  neigh_clusters <- neigh_clusters[neigh_clusters != df$cluster_spatial[node_idx[1]]]
  # intersect with large cluster ids
  adjacent_large <- neigh_clusters[neigh_clusters %in% large_ids]
  return(adjacent_large)
}

# For each small cluster, prefer assigning to an adjacent large cluster 
for (sc in small_clusters) {
  idx <- which(df$cluster_spatial == sc)
  if (length(idx) == 0) next
  
  # try to find adjacent large clusters via kNN edges
  adjacent_large <- get_adjacent_large(idx, nn$nn.idx, df, large_clusters)
  
  chosen_large <- NA_integer_
  if (length(adjacent_large) > 0) {
    # count adjacency strength 
    adj_counts <- sapply(adjacent_large, function(lid) {
      # neighbours of small nodes
      neighs <- as.vector(nn$nn.idx[idx, , drop = FALSE])
      sum(df$cluster_spatial[neighs] == lid, na.rm = TRUE)
    })
    # choose the adjacent large cluster with highest adjacency
    best_idx <- which(adj_counts == max(adj_counts))
    candidates <- adjacent_large[best_idx]
    if (length(candidates) == 1) {
      chosen_large <- candidates
    } else { # if the candidate size is greater than 1
      sc_centroid <- colMeans(coords[idx, , drop = FALSE])
      cand_centroids <- large_centroids %>% filter(cluster_spatial %in% candidates)
      dists <- sqrt((cand_centroids$cx - sc_centroid[1])^2 + (cand_centroids$cy - sc_centroid[2])^2)
      chosen_large <- candidates[which.min(dists)]
    }
  } else {
    # choose nearest large centroid 
    sc_centroid <- colMeans(coords[idx, , drop = FALSE])
    if (nrow(large_centroids) == 0) {
      # no large clusters exist
      next
    }
    dists <- sqrt((large_centroids$cx - sc_centroid[1])^2 + (large_centroids$cy - sc_centroid[2])^2)
    chosen_large <- large_centroids$cluster_spatial[which.min(dists)]
  }
  
  # reassign small cluster nodes to chosen large cluster id
  df$cluster_spatial[idx] <- chosen_large
}


# ensure every label corresponds to a connected component
# this splits any label that is still disconnected after merging

new_label <- 1L
final_labels <- integer(nrow(df))
for (lab in sort(unique(df$cluster_spatial))) {
  idx <- which(df$cluster_spatial == lab)
  if (length(idx) == 0) next
  subg <- induced_subgraph(g, idx)
  comps <- components(subg)$membership
  for (comp_id in unique(comps)) {
    local_pos <- which(comps == comp_id)
    nodes <- idx[local_pos]
    final_labels[nodes] <- new_label
    new_label <- new_label + 1L
  }
}
df$cluster_spatial <- final_labels

# Plot
distinctColorPalette <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
n_clusters <- length(unique(df$cluster_spatial))

ggplot(df, aes(x = x, y = y, color = factor(cluster_spatial))) +
  geom_point(size = 0.6, alpha = 0.9) +
  scale_color_manual(values = distinctColorPalette(n_clusters)) +
  coord_equal() +
  theme_minimal() +
  ggtitle("Spatial Clusters (contiguity enforced)")


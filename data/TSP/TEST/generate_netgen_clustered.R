library(netgen)

clusters <- c(4L, 5L, 6L, 7L, 8L)
n_instances <- 1L
output_dir <- "./cluster_netgen"
dir.create(file.path(output_dir), showWarnings = FALSE, recursive = TRUE)

ii = 0L
for(idx in seq_along(clusters)) {
  cluster <- clusters[idx]
  for(i in 0:(n_instances - 1)) {
    x = generateClusteredNetwork(n.cluster = cluster, n.points = 500L)
    file_path = file.path(output_dir, sprintf("%03d.tsp", ii + i))
    exportToTSPlibFormat(x, file_path, name = "cluster_netgen", use.extended.format = FALSE, digits = 5)
  }
  ii = ii + n_instances
}

library(netgen)

set.seed(0)

clusters <- c(4L, 5L, 6L, 7L, 8L)
n_instances <- 40L
output_dir <- "./cluster_netgen"
dir.create(file.path(output_dir), showWarnings = FALSE, recursive = TRUE)

ii = 0L
for(idx in seq_along(clusters)) {
  cluster <- clusters[idx]
  for(i in 0:(n_instances - 1)) {
    sigmas = lapply(1:cluster, function(x) diag(runif(1, min = 2000000000, max = 4000000000), nrow = 2, ncol = 2))
    n.points = sample(50:600, 1)
    x = generateClusteredNetwork(n.cluster = cluster, n.points = n.points, lower = 0, upper = 999999, sigmas=sigmas)
    file_path = file.path(output_dir, sprintf("%03d.tsp", ii + i))
    exportToTSPlibFormat(x, file_path, name = "cluster_netgen", use.extended.format = FALSE, digits = 0)
  }
  ii = ii + n_instances
}

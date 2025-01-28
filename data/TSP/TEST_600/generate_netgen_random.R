library(netgen)

n_instances <- 5L
output_dir <- "./uniform_portgen"
dir.create(file.path(output_dir), showWarnings = FALSE, recursive = TRUE)

for(i in 0:(n_instances - 1)) {
  x = generateRandomNetwork(n.points = 600L, upper = 999999)
  file_path = file.path(output_dir, sprintf("%03d.tsp", i))
  exportToTSPlibFormat(x, file_path, name = "uniform_portgen", use.extended.format = FALSE, digits = 0)
}

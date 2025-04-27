library(netgen)

n_instances <- 200L
output_dir <- "./uniform_portgen"
dir.create(file.path(output_dir), showWarnings = FALSE, recursive = TRUE)

for(i in 0:(n_instances - 1)) {
  n.points = sample(50:600, 1)
  x = generateRandomNetwork(n.points = n.points, upper = 999999)
  file_path = file.path(output_dir, sprintf("%03d.tsp", i))
  exportToTSPlibFormat(x, file_path, name = "uniform_portgen", use.extended.format = FALSE, digits = 0)
}

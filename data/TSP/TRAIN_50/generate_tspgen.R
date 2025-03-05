library(tspgen)
library(netgen)

mutators <- c("doCompressionMutation", "doExpansionMutation", "doExplosionMutation", "doGridMutation")
dirs <- c("compression", "expansion", "explosion", "grid")
n_instances <- 200L
output_dir <- "."

for(idx in seq_along(mutators)) {
  mutator <- mutators[idx]
  dir_name <- dirs[idx]
  for(i in 0:(n_instances - 1)) {
    collection = init()
    collection = addMutator(collection, mutator)
    x = build(n = 50L, iters = 10L, upper = 999999, collection = collection, bound.handling = "uniform")
    dir.create(file.path(output_dir, dir_name), showWarnings = FALSE, recursive = TRUE)
    file_path = file.path(output_dir, dir_name, sprintf("%03d.tsp", i))
    exportToTSPlibFormat(x, file_path, name = dir_name, use.extended.format = FALSE, digits = 0)
  }
}

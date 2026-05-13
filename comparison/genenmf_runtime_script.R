library(anndataR)
library(rhdf5)

library(Seurat)
library(ggplot2)
library(patchwork)
library(tidyr)
library(dplyr)
library(RColorBrewer)
library(viridis)
library(GeneNMF)
library(msigdbr)
library(fgsea)
library(UCell)


library(anndataR)
library(Matrix)
library(Seurat)
library(GeneNMF)

# Paths to datasets
dataset_files <- c(
  "../data/gbmap_downsample_10.h5ad",
  "../data/gbmap_downsample_25.h5ad",
  "../data/gbmap_downsample_50.h5ad",
  "../data/gbmap_downsample_75.h5ad"
)

# Storage for runtime results
results <- data.frame(
  dataset = character(),
  n_cells = numeric(),
  runtime_sec = numeric(),
  stringsAsFactors = FALSE
)

for (file in dataset_files) {
  
  dataset_name <- tools::file_path_sans_ext(basename(file))
  cat("Processing dataset:", dataset_name, "\n")
  
  # ---- Load AnnData ----
  adata <- read_h5ad(file)

  # sparse matrix
  X <- Matrix::Matrix(adata$X, sparse = TRUE)

  # transpose to genes x cells
  X <- Matrix::t(X)

  meta <- as.data.frame(adata$obs)

  # ensure cell names match
  colnames(X) <- rownames(meta)

  # create Seurat object
  seu <- CreateSeuratObject(
    counts = X,
    meta.data = meta
  )

  # store normalized matrix in the data layer (Seurat v5)
  LayerData(seu, assay = "RNA", layer = "data") <- X

  DefaultAssay(seu) <- "RNA"

  # Split by donor
  seu.list <- SplitObject(seu, split.by = "donor_id")
  
  # ---- Start timing ----
  start_time <- Sys.time()
  
  geneNMF.programs <- multiNMF(
    seu.list,
    assay = "RNA",
    k = 4:9,
    min.exp = 0.05
  )
  
  # ---- Stop timing ----
  elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  cat("Finished", dataset_name, "in", elapsed_time, "seconds\n")
  
  # Store results
  results <- rbind(
    results,
    data.frame(
      dataset = dataset_name,
      n_cells = ncol(seu),
      runtime_sec = elapsed_time
    )
  )
}

# Save CSV
csv_output <- "../results/runtime_over_cells.csv"

dir.create(dirname(csv_output), recursive = TRUE, showWarnings = FALSE)
write.csv(results, csv_output, row.names = FALSE)

cat("Saved runtime CSV to", csv_output, "\n")
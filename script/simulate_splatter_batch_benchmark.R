require(splatter)
require(scater)
require(ggplot2)
seed <- 0
set.seed(seed)

output_dirs_names <- c("simul1_dropout_005_b1_500_b2_900", 
                      "simul2_dropout_025_b1_500_b2_900", 
                      "simul3_dropout_005_b1_500_b2_450", 
                      "simul4_dropout_025_b1_500_b2_450", 
                      "simul5_dropout_005_b1_80_b2_400", 
                      "simul6_dropout_025_b1_80_b2_400")
parameters <- as.data.frame(t(data.frame(p1=c(dropout = 0.05,
                     batchCells = c(500,900),
                     nGroups = 2,
                     nGenes = 5000,
                     group.prob = c(0.3, 0.7),
                     de.prob = c(0.2, 0.1),
                     de.downProb = c(0.3, 0.4),
                     de.facLoc = 0.5,
                     de.facScale = 0.2), 
                p2=c(dropout = 0.25,
                     batchCells = c(500,900),
                     nGroups = 2,
                     nGenes = 5000,
                     group.prob = c(0.3, 0.7),
                     de.prob = c(0.2, 0.1),
                     de.downProb = c(0.3, 0.4),
                     de.facLoc = 0.5,
                     de.facScale = 0.2),
                p3=c(dropout = 0.05,
                     batchCells = c(500,450),
                     nGroups = 2,
                     nGenes = 5000,
                     group.prob = c(0.3, 0.7),
                     de.prob = c(0.2, 0.1),
                     de.downProb = c(0.3, 0.4),
                     de.facLoc = 0.5,
                     de.facScale = 0.2),
                p4=c(dropout = 0.25,
                     batchCells = c(500,450),
                     nGroups = 2,
                     nGenes = 5000,
                     group.prob = c(0.3, 0.7),
                     de.prob = c(0.2, 0.1),
                     de.downProb = c(0.3, 0.4),
                     de.facLoc = 0.5,
                     de.facScale = 0.2),
                p5=c(dropout = 0.05,
                     batchCells = c(80,400),
                     nGroups = 2,
                     nGenes = 5000,
                     group.prob = c(0.3, 0.7),
                     de.prob = c(0.2, 0.1),
                     de.downProb = c(0.3, 0.4),
                     de.facLoc = 0.5,
                     de.facScale = 0.2),
                p6=c(dropout = 0.25,
                     batchCells = c(80,400),
                     nGroups = 2,
                     nGenes = 5000,
                     group.prob = c(0.3, 0.7),
                     de.prob = c(0.2, 0.1),
                     de.downProb = c(0.3, 0.4),
                     de.facLoc = 0.5,
                     de.facScale = 0.2)
                )))

prefix <- "splatter_benchmark_"

for (simu_idx in 1:6) {

    unique_affix <- output_dirs_names[simu_idx]
    cell_in_each_batch <- c(parameters$batchCells1[simu_idx], 
                            parameters$batchCells2[simu_idx])
    cell_in_each_cell_type_proba <- c(parameters$group.prob1[simu_idx],
                                      parameters$group.prob2[simu_idx])
    
    params <- newSplatParams()
    params <- setParam(params, "dropout.mid", parameters$dropout[simu_idx])
    params <- setParam(params, "batchCells", cell_in_each_batch)
    params <- setParam(params, "nGenes", parameters$nGenes[simu_idx])
    params <- setParam(params, "group.prob", cell_in_each_cell_type_proba)
    params <- setParam(params, "de.prob", c(parameters$de.prob1[simu_idx], 
                                            parameters$de.prob2[simu_idx]))
    params <- setParam(params, "de.downProb", c(parameters$de.downProb1[simu_idx], 
                                                parameters$de.downProb2[simu_idx]))
    params <- setParam(params, "de.facLoc", parameters$de.facLoc[simu_idx])
    params <- setParam(params, "de.facScale", parameters$de.facScale[simu_idx])
    #splatSimDE
    sim <- splatSimulate(params, method = "group", verbose = FALSE)
    # sim <- addGeneLengths(sim)
    # tpm(sim) <- calculateTPM(sim, rowData(sim)$Length)
    # fpkm(sim) <- calculateFPKM(sim, rowData(sim)$Length)
    data_dir <- "F:/OneDrive - Hong Kong Baptist University/year1_1/cgi_datasets"
    # data_dir <- "E:/OneDrive - stu.hit.edu.cn/year1_1/cgi_datasets"
    dir.create(file.path(data_dir, paste0(prefix, unique_affix)), showWarnings = FALSE)
    write.csv(colData(sim), file.path(data_dir, paste0(prefix, unique_affix), "cells_info.csv"))
    write.csv(counts(sim), file.path(data_dir, paste0(prefix, unique_affix), "counts.csv"))
    # write.csv(tpm(sim), file.path(data_dir, paste0(prefix, unique_affix), "tpm.csv"))
    # write.csv(fpkm(sim), file.path(data_dir, paste0(prefix, unique_affix), "fpkm.csv"))
    
    write.csv(assays(sim)$TrueCounts, file.path(data_dir, paste0(prefix, unique_affix), "true_counts.csv"))
    
}
# sim <- normalize(sim)
# plotPCA(sim, shape_by = "Batch", colour_by = "Group")
datasets_names <- c()
# for (group_num in 2:10) {
#     datasets_names <- c(datasets_names, paste0('"', prefix, group_num, '"'))
# }
# for (dropout_mid in 11:2) {
#     datasets_names <- c(datasets_names, paste0('"', prefix, dropout_mid, '"'))
# }
# for (batch in 2:5) {
#     datasets_names <- c(datasets_names, paste0('"', prefix, batch, '"'))
# }

# for (batch in 2) {
#     datasets_names <- c(datasets_names, paste0('"', prefix, batch, '"'))
# }
for (output_dir_name in output_dirs_names) {
    datasets_names <- c(datasets_names, paste0('"', prefix, output_dir_name, '"'))
}
writeLines(paste(datasets_names, collapse = ", "), file.path(data_dir, "splatter", paste0(prefix, ".txt")))

# # generate batch effect data if batch effect exists
# sim1 <- splatSimulate(params, batch.rmEffect = FALSE, method = "group",verbose = FALSE)
# # generate no batch effect data (ground truth)
# sim2 <- splatSimulate(params, batch.rmEffect = TRUE, method = "group", verbose = FALSE)

# sce <- mockSCE()
# params <- splatEstimate(sce)
# sim <- splatSimulate(params)
# 
# params <- newSplatParams()
# params <- setParam(params, "nGenes", 5000)
# params <- setParams(params, update = list(nGenes = 8000, mean.rate = 0.5))
# getParams(params, c("nGenes", "mean.rate", "mean.shape"))
# params <- setParams(params, mean.shape = 0.5, de.prob = 0.2)
# params <- newSplatParams(lib.loc = 12, lib.scale = 0.6)
# counts <- counts(sce)
# params <- splatEstimate(counts)
# sim <- splatSimulate(params, nGenes = 1000)
# counts(sim)[1:5, 1:5]
# head(rowData(sim))
# head(colData(sim))
# names(assays(sim))
# assays(sim)$CellMeans[1:5, 1:5]
# sim <- normalize(sim)
# plotPCA(sim)
# 
# sim.groups <- splatSimulate(group.prob = c(0.5, 0.5), method = "groups",
#                             verbose = FALSE)
# sim.groups <- normalize(sim.groups)
# plotPCA(sim.groups, colour_by = "Group")
# 
# sim.paths <- splatSimulate(method = "paths", verbose = FALSE)
# sim.paths <- normalize(sim.paths)
# plotPCA(sim.paths, colour_by = "Step")
# 
# sim.batches <- splatSimulate(batchCells = c(50, 50), verbose = FALSE)
# sim.batches <- normalize(sim.batches)
# plotPCA(sim.batches, colour_by = "Batch")
# 
# sim.groups <- splatSimulate(batchCells = c(50, 50), group.prob = c(0.5, 0.5),
#                             method = "groups", verbose = FALSE)
# sim.groups <- normalize(sim.groups)
# plotPCA(sim.groups, shape_by = "Batch", colour_by = "Group")
# 
# listSims()
# 
# sim <- simpleSimulate(verbose = FALSE)
# sim <- addGeneLengths(sim)
# head(rowData(sim))
# 
# tpm(sim) <- calculateTPM(sim, rowData(sim)$Length)
# tpm(sim)[1:5, 1:5]
# 
# sim1 <- splatSimulate(nGenes = 1000, batchCells = 20, verbose = FALSE)
# sim2 <- simpleSimulate(nGenes = 1000, nCells = 20, verbose = FALSE)
# comparison <- compareSCEs(list(Splat = sim1, Simple = sim2))
# 
# names(comparison)
# names(comparison$Plots)
# comparison$Plots$Means
# 
# ggplot(comparison$ColData, aes(x = sum, y = detected, colour = Dataset)) +
#     geom_point()
# 
# difference <- diffSCEs(list(Splat = sim1, Simple = sim2), ref = "Simple")
# difference$Plots$Means
# difference$QQPlots$Means
# 
# 
# 

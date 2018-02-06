library(org.Hs.eg.db)
library(annotate)

args = commandArgs(trailingOnly=TRUE)
file <- read.table(file=args[2], sep = "\t", header = TRUE)
entrezGeneIds <- as.character(file[,1])
n <- getSYMBOL(entrezGeneIds, data='org.Hs.eg')
write.table(n, file=args[1], quote=FALSE, sep='\t', col.names = NA)

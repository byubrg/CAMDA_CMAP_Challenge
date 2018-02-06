#! /bin/bash

#install all the R packages in the environment
#source activate WishBuilderDependencies 
conda install -y -c bioconda r-sleuth 
conda install -y -c r r-xml=3.98_1.5
conda install -y gcc
Rscript installRPackages.R
#source deactivate WishBuilderDependencies

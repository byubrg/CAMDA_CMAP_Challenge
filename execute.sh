#! /bin/bash

#Make Folder
ReformatingData=ReformatingData
Data=Data
CMAP_2_Entrez=$ReformatingData/CMAP_2_Entrez
scikitLearnAlgorithms=ScikitLearnAlgorithms

#Given Files 
camdaGiven=$Data/CAMDA_Challange_dataset_filenames.txt  
cmapExpressionData=$CMAP_2_Entrez/data.tsv.gz  
cmapMetadataInfo=$CMAP_2_Entrez/metadata.tsv.gz

#Created Files
trainingReformatedCamda=$Data/trainingReformatedCamda.txt.gz
testReformatedCamda=$Data/testReformatedCamda.txt.gz 
trainingReformatedCamdaPC3=$Data/trainingReformatedCamda_PC3.txt.gz
trainingReformatedCamdaMCF7=$Data/trainingReformatedCamda_MCF7.txt.gz
testReformatedCamdaPC3=$Data/testReformatedCamda_PC3.txt.gz
testReformatedCamdaMCF7=$Data/testReformatedCamda_MCF7.txt.gz
discretePredictionsOut=discretePredictionsOut.csv

#Download and filter CMAP scripts 
reformatCamda=$ReformatingData/reformatCamdaGivin.py
cmapInstall=$CMAP_2_Entrez/"install.sh"
cmapDownload=$CMAP_2_Entrez/download.sh
cmapParse=$CMAP_2_Entrez/parse.sh

#Algorithm Scripts
#execute=$scikitLearnAlgorithms/execute.py
execute=execute.py

#Opening environment for dependencies
minicondaBin=Software/miniconda/bin/
cd $minicondaBin
source activate scikitLearn_env
cd ../../..

#Download CMAP data and reformat
#bash $cmapInstall
#bash $cmapDownload
#bash $cmapParse 

#Filter the dataset for the camda challenge
#python3 $reformatCamda $camdaGiven $cmapExpressionData $trainingReformatedCamda $testReformatedCamda $cmapMetadataInfo


##The This script sets up cross validation and executes each algorithm in collaberation of ensemble methods 
python3 $execute $trainingReformatedCamdaPC3 $trainingReformatedCamdaMCF7 $testReformatedCamdaPC3 $testReformatedCamdaMCF7 $discretePredictionsOut


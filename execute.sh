#! /bin/bash

#Make Folder
ReformatingData=ReformatingData
Data=Data
CMAP_2_Entrez=$ReformatingData/CMAP_2_Entrez

#Given Files 
camdaGiven=$Data/CAMDA_Challange_dataset_filenames.txt  
cmapExpressionData=$CMAP_2_Entrez/data.tsv.gz  
cmapMetadataInfo=$CMAP_2_Entrez/metadata.tsv.gz

#Created Files
trainingReformatedCamda=$Data/trainingReformatedCamda.txt.gz
testReformatedCamda=$Data/testReformatedCamda.txt.gz 

#Download and filter CMAP scripts 
reformatCamda=$ReformatingData/reformatCamdaGivin.py
cmapInstall=$CMAP_2_Entrez/"install.sh"
cmapDownload=$CMAP_2_Entrez/download.sh
cmapParse=$CMAP_2_Entrez/parse.sh

#Algorithm Scripts
mlp=$scikitLearnAlgorithms/mlp.py

#Opening environment for dependencies
minicondaBin=Software/miniconda/bin/
cd $minicondaBin
source activate scikitLearn_env
cd ../../..

#Download CMAP data and reformat
bash $cmapInstall
bash $cmapDownload
bash $cmapParse 

#Filter the dataset for the camda challenge
#python3 $reformatCamda $camdaGiven $cmapExpressionData $trainingReformatedCamda $testReformatedCamda $cmapMetadataInfo

#The scilearn algorithms
python3 $mlp $trainingReformatedCamda

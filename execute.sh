#! /bin/bash

#Make Folder
Data=Data
mkdir -p $Data

#Given Files 
camdaGiven=$Data/CAMDA_Challange_dataset_filenames.txt  
cmapExpressionData=$Data/data.tsv.gz  
cmapMetadataInfo=$Data/metadata.tsv.gz

#Created Files
trainingReformatedCamda=$Data/trainingReformatedCamda.txt.gz
testReformatedCamda=$Data/testReformatedCamda.txt.gz 

#Scripts
reformatCamda=reformatCamdaGivin.py
mlp=mlp.py

#environment
minicondaBin=Software/miniconda/bin/
cd $minicondaBin
source activate skikitLearn_env
cd ../../..

#python3 $reformatCamda $camdaGiven $cmapExpressionData $trainingReformatedCamda $testReformatedCamda $cmapMetadataInfo
python3 $mlp $trainingReformatedCamda

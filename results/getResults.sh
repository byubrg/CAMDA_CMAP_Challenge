#! /bin/bash

ensemble="ensembleMethodSolutions.csv"
rf="RFSolutions.csv"
logReg="logRegsolutions.csv"
actual="CAMDA_Challange_dataset_filenames_complete_04_23_18.csv"

python3 getResults.py $ensemble $rf $logReg $actual

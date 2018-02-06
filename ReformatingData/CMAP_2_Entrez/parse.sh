#! /bin/bash

#Folders used
redirectedTmpFolder=tmp
softwareFolder=Software
parseFolder=parse_helpers

#Parse Files
makeHeaders=$parseFolder/makeHeaders.py
makeBS=$parseFolder/makeBashScript.py
entrezIdConvert=$parseFolder/entrezIdConvert.R

#Files I will create
fileNames=$redirectedTmpFolder/"fileNames.txt"
bashFileNames=$redirectedTmpFolder/"bashFileNames.sh"
entrezIdConvertBash=$redirectedTmpFolder/"entrezIdConvert.sh"
convertedHeaders=$redirectedTmpFolder/"convertedHeaders.tsv"
CMAPLocations=$redirectedTmpFolder/CMap_SCAN_EntrezGene

#Downloaded and Installed in install.sh
minicondaPath=$softwareFolder/miniconda/bin/

#outFiles
expressionOut=data.tsv.gz
metadataOut=metadata.tsv.gz

rm -f $metadataOut
rm -f $expressionOut

#setting up java environment to read gene symbols from Entrez Gene
echo "Setting up environment"
#source activate WishBuilderDependencies

#grab all the file names from CMAP_SCAN_EntrezGene
ls tmp/CMap_SCAN_EntrezGene > $fileNames

#Make a bash script to execute the R script with the first fileName to get the headers. Note that previous script was ran to ensure consistency of row names
echo "#! /bin/bash" > $entrezIdConvertBash 
echo -n "Rscript \"$entrezIdConvert\" \"$convertedHeaders\" $CMAPLocations/" >> $entrezIdConvertBash 
sed -n 1p $fileNames >> $entrezIdConvertBash

bash $entrezIdConvertBash

#Writes out headers and makes a bash script to execute the parse script for all the files
python3 $makeHeaders $convertedHeaders $expressionOut $metadataOut
python3 $makeBS $fileNames $bashFileNames $expressionOut $metadataOut

bash $bashFileNames

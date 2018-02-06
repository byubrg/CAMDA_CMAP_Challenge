#! /bin/bash

#make Software Directory
softwareFolder=Software
mkdir -p $softwareFolder

#installing miniconda
softwareName=$softwareFolder/miniconda
url="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
#url="https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
fileName="$softwareFolder/$(basename $url)"

curl -o "$fileName" -L "$url"
bash "$fileName" -b -p $softwareName
rm "$fileName"

#setting up environment for this project
export PATH=$softwareName/bin:$PATH
conda create --name skikitLearn_env -y python=3.6.2 NumPy=1.13.3 SciPy=1.0.0 scikit-learn=0.19.1


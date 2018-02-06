#make redirectedTempFolder
redirectedTempFolder=tmp
mkdir -p $redirectedTempFolder

#downloading the tar file 
url="https://osf.io/vd7cf/download"
fileName=$redirectedTempFolder/cmap.tar

wget -O $fileName $url

cd $redirectedTempFolder
tar -xvf cmap.tar

rm cmap.tar


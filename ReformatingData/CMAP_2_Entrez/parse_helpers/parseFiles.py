import sys, gzip

patientFile = sys.argv[1]
expressionOut=sys.argv[2]
metadataOut=sys.argv[3]

with open(patientFile, 'r') as f :
    with gzip.open(expressionOut, 'a') as exO :
        with gzip.open(metadataOut, 'a') as metaO :
            firstLineList = f.readline().strip('\n').split('\t')[1].split('__')
            if (firstLineList[0] == "Perturbed") :
                try :
                    metaO.write((firstLineList[7] + "\t" + "Perturbagen" + "\t" + firstLineList[2] + "\n").encode())
                    metaO.write((firstLineList[7] + "\t" + "Concentration" + "\t" + firstLineList[3] + "\n").encode())
                    metaO.write((firstLineList[7] + "\t" + "Cell Line Name" + "\t" + firstLineList[4] + "\n").encode())
                    metaO.write((firstLineList[7] + "\t" + "Batch" + "\t" + firstLineList[5][5:] + "\n").encode())
                    metaO.write((firstLineList[7] + "\t" + "Array" + "\t" + firstLineList[6] + "\n").encode())
                    exO.write((firstLineList[7]).encode()) 
                except IndexError :
                    pertubagen = "_".join(firstLineList[1].split('_')[1:])
                    metaO.write((firstLineList[6] + "\t" + "Perturbagen" + "\t" + pertubagen + "\n").encode())
                    metaO.write((firstLineList[6] + "\t" + "Concentration" + "\t" + firstLineList[2] + "\n").encode())
                    metaO.write((firstLineList[6] + "\t" + "Cell Line Name" + "\t" + firstLineList[3] + "\n").encode())
                    metaO.write((firstLineList[6] + "\t" + "Batch" + "\t" + firstLineList[4][5:] + "\n").encode())
                    metaO.write((firstLineList[6] + "\t" + "Array" + "\t" + firstLineList[5] + "\n").encode())
                    exO.write((firstLineList[6]).encode()) 
            else :
                metaO.write((firstLineList[4] + "\t" + "Perturbagen" + "\t" + firstLineList[0] + "\n").encode())
                metaO.write((firstLineList[4] + "\t" + "Cell Line Name" + "\t" + firstLineList[1] + "\n").encode())
                metaO.write((firstLineList[4] + "\t" + "Batch" + "\t" + firstLineList[2][5:] + "\n").encode())
                metaO.write((firstLineList[4] + "\t" + "Array" + "\t" + firstLineList[3] + "\n").encode())
                exO.write((firstLineList[4]).encode()) 
                
            
            for line in f :
                line = line.strip('\n').split('\t') 
                exO.write(("\t" + line[1]).encode())
            exO.write(("\n").encode())

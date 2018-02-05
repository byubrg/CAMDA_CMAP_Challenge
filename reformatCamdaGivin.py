import sys, gzip
import numpy as np

camdaGiven = sys.argv[1]
cmapData = sys.argv[2]
trainingOut = sys.argv[3]
testOut = sys.argv[4]
cmapMeta = sys.argv[5]


with gzip.open(cmapData, 'r') as cmapIn :
    data = np.genfromtxt(cmapIn,delimiter='\t',dtype=str)

##Creating a dictionary for each of the samples to it's index number
sampleToIndex = {}
index = 0
for element in  data[:,0] :
    sampleToIndex[element] = index
    index += 1

##Make a directory with all the possible extensions and pertubagen type to list of sample Id
extenPertToSample = {}
sampleToPerturbagen = {}
with gzip.open(cmapMeta, 'r') as metaIn :
    metaIn.readline()
    for line in metaIn :
        lineList = line.decode().strip('\n').split('\t')
        if lineList[1] == "Cell Line Name" :
            if lineList[2] == "MCF7" or lineList[2] == "PC3" : 
                sampleKey = lineList[0].split(".")[-1] + "_" + lineList[2] + "_" + lineList[0].split(".")[0]
                if sampleKey not in extenPertToSample :
                    extenPertToSample[sampleKey] = []
                extenPertToSample[sampleKey].append(lineList[0])
        if lineList[1] == "Perturbagen" :
            sampleToPerturbagen[lineList[0]] = lineList[2]
                
with open(camdaGiven, 'r') as camdaIn : 
    with gzip.open(trainingOut, 'w') as trainingO :
        with gzip.open(testOut, 'w') as testO :
            ##Skip the beginning lines in the camdaFile
            while True :
                lineList = camdaIn.readline().strip('\n').split('\t')
                if 'S' in lineList[0] :
                    break
            firstLine = "S No.\tLiver\tDrug"
            
            ##Make the first line of both files
            for element in data[0,1:] :
                firstLine += "\tMCF7_" + element
            for element in data[0,1:] :
                firstLine += "\tPC3_" + element

            firstLine += "\n"

            ##Writing Out the created headers
            trainingO.write((firstLine).encode())
            testO.write((firstLine).encode())

            ##Write out info that can be accepted by Micheal's code
            for line in camdaIn :
                 print(line)
                 lineList = line.strip('\n').split('\t')
                 ##Focus first on mcf7
                 if lineList[3][0] == "'" :
                     lineList[3] = lineList[3][1:]

                 mcf7Compound = np.array(data[sampleToIndex[lineList[3]],1:], 'float')
                 
                 mcf7Perturbagen = sampleToPerturbagen[lineList[3]]
                 ##Find All MCF7 vehicle expression values for patient
                 extenList = lineList[4].split(".")
                 if len(extenList) > 2 :

                     allMCF7VehicleData = []
                     for exten in extenList :
                         if exten != "" :
                             sampleLists = extenPertToSample[str(exten + "_MCF7_" + lineList[3].split(".")[0])]
                             for sample in sampleLists :
                                 allMCF7VehicleData.append(np.array(data[sampleToIndex[sample],1:], dtype='float'))
                             
                     mcf7Vehicle = np.mean(np.array(allMCF7VehicleData), axis=0)
                 
                     mcf7Result = np.subtract(mcf7Compound, mcf7Vehicle)
                 else :
                     if lineList[4][0] == "'" :
                         lineList[4] = lineList[4][1:]
                     mcf7Result = np.array(data[sampleToIndex[lineList[4]],1:], dtype='float')

                 ##Now focusing on PC3
                 if lineList[5][0] == "'" :
                     lineList[5] = lineList[5][1:]

                 PC3Compound = np.array(data[sampleToIndex[lineList[5]],1:], 'float')
                          
                 PC3Perturbagen = sampleToPerturbagen[lineList[5]]

                 ##Find All PC3 vehicle expression values for patient
                 extenList = lineList[6].split(".")
                 if len(extenList) > 2 :
                     allPC3VehicleData = []
                     for exten in extenList :
                         if exten != "" :
                             sampleLists = extenPertToSample[str(exten + "_PC3_" + lineList[5].split(".")[0])]
                             for sample in sampleLists :
                                 allPC3VehicleData.append(np.array(data[sampleToIndex[sample],1:], dtype='float'))
                             
                     PC3Vehicle = np.mean(np.array(allPC3VehicleData), axis=0)

                     PC3Result = np.subtract(PC3Compound, PC3Vehicle)
                 else :
                     if lineList[6][0] == "'" :
                         lineList[6] = lineList[6][1:]
                     PC3Result = np.array(data[sampleToIndex[lineList[6]],1:], dtype='float')
                 
                 assert mcf7Perturbagen == PC3Perturbagen 
                     
                 if "T" in lineList[1] :
                     trainingO.write((lineList[0] + "\t" + lineList[2] + "\t" +  mcf7Perturbagen + "\t" + "\t".join(mcf7Result.astype(str)) + "\t" + "\t".join(PC3Result.astype(str))+ "\n").encode()) 
                 else : 
                     testO.write((lineList[0] + "\tNA\t" + mcf7Perturbagen + "\t" + "\t".join(mcf7Result.astype(str)) + "\t" + "\t".join(PC3Result.astype(str))+ "\n").encode()) 

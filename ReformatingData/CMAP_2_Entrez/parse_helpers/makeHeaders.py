import sys, gzip


headers = sys.argv[1]
dataOut = sys.argv[2]
metadataOut = sys.argv[3]

with open(headers, 'r') as f :
    with gzip.open(dataOut, 'w') as dO :
        f.readline()
        dO.write(("Sample").encode())
        for line in f :
            lineList = line.strip('\n').split('\t')
            if lineList[1] == "NA" :
                dO.write(("\tLOC" + lineList[0]).encode())
            else :
                dO.write(("\t" + lineList[1]).encode())
        dO.write(("\n").encode())

with gzip.open(metadataOut, 'w') as f :
    f.write(("Sample\tVariable\tValue\n").encode())

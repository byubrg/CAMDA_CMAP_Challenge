import sys

inFile = sys.argv[1]
outFile = sys.argv[2]
expressionOut = sys.argv[3]
metadataOut = sys.argv[4]

with open(inFile, 'r') as f :
    with open(outFile, 'w') as o :
        o.write("#! /bin/bash\n")
        i = 0
        for line in f :
            i = i + 1
            line = line.strip('\n')
            o.write("python3 \"parse_helpers/parseFiles.py\" \"tmp/CMap_SCAN_EntrezGene/" + line + "\" " + expressionOut + " " + metadataOut + "\n")
            if i % 50 == 0 :
                o.write("echo \"" + str(i) + " of 7057\"\n")

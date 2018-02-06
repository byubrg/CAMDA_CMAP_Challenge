import sys, gzip

file = sys.argv[1]

uniqueDrugs = set()
with gzip.open(file, 'r') as f:
    for line in f :
#         lineList = line.decode().strip('\n').split('\t')
#         uniqueDrugs.add(lineList[2])
        print(str(len(line.decode().strip('\n').split('\t'))))

#print(str(len(uniqueDrugs)))

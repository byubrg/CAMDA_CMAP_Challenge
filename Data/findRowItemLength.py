import sys, gzip

file = sys.argv[1]

with gzip.open(file, 'r') as f:
    for line in f :
        print(str(len(line.decode().strip('\n').split('\t'))))

import sys,gzip

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]

with gzip.open(file1, 'r') as f:
    with gzip.open(file2, 'w') as of1 :
        with gzip.open(file3, 'w') as of2 :
            for line in f :
                lineList = line.decode().strip('\n').split('\t')
                of1.write(("\t".join(lineList[:12082]) + "\n").encode())
                of2.write(("\t".join(lineList[:2] + lineList[12082:]) + "\n").encode())


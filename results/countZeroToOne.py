import sys

solutionsFile = sys.argv[1]

with open(solutionsFile,'r') as f :
    f.readline()
    f.readline()
    numSolutions = 0
    numOnes = 0
    for line in f :
        lineList = line.strip('\n').split(',')
        numSolutions += 1
        numOnes += int(lineList[2])
    print("number of Solutions: ", numSolutions)
    print("number of ones: ", numOnes)
        

"""WordCounter.py"""
# Takes the basic Spark Word Count example a little further to find the most 
# common words in each of a list of input files and also in the set of files
# as a whole.
# Created by Lisa Gaudette as a simple learning project for both Spark and Python
from pyspark import SparkContext
import shutil
import json
import argparse, sys
import string, re

stopwords = []
letterPattern = re.compile('[^a-z^ ]+')

def tokenizer(inputString):
    global letterPattern
    onlyLettersAndWhitespace = letterPattern.sub(' ', inputString.lower())
    return onlyLettersAndWhitespace.split(" ")
   
def stopwordFilter(inputString):
    return inputString[0].strip() not in stopwords

def createStopwordList(stopFile):
    global stopwords
    stopwords.append("")
    f = open(stopFile, 'r')
    for line in f:
        if line.strip().startswith("#"):
            continue
        else:
            stopwords.append(line.strip())
    f.close()
    stopwords.sort()
    

def main(argv):
    parser = argparse.ArgumentParser(description="Count words in files")
    parser.add_argument("files", metavar="F", nargs='+', help="A file to count words in")
    parser.add_argument("-s", "--stopFile", help="a file containing words to ignore, one word per line, ignores lines starting with #")
    parser.add_argument("-o", "--outputFile", help="the file to store the output in")
    args = parser.parse_args()    
    
    shutil.rmtree(args.outputFile, ignore_errors=True)

    sc = SparkContext("local", "WordCounter")    
    createStopwordList(args.stopFile)
    documentCounts = []
    output = []
    for file in args.files:        
        fileLines = sc.textFile(file)
        documentCounts.append(fileLines.flatMap(tokenizer) \
                .map(lambda word: (word, 1)) \
                .reduceByKey(lambda a, b: a + b) \
                .filter(stopwordFilter)
                .map(lambda (a, b): (b, a) ) \
                .sortByKey(False, 1) \
                .map(lambda (a, b): (b, a)))
        output.append({'name': file.split(".")[0], 'wordCounts': []})                 

    # Combine the word counts for the documents      
    combinedCounts = sc.union(documentCounts) \
                .reduceByKey(lambda a, b: a + b) \
                .map(lambda (a, b): (b, a) ) \
                .sortByKey(False, 1) \
                .map(lambda (a, b): (b, a))
                
    # Write most common words to file in JSON format
    f = open(args.outputFile, 'w')
    countIndex = 0;
    for counts in documentCounts:
        output[countIndex]['wordCounts'] = counts.take(25)
        countIndex = countIndex + 1
        
    output.append({'name' : "Combined", 'wordCounts' : combinedCounts.take(25)})
    json.dump(output,f)
    f.close()

if __name__ == "__main__":
   main(sys.argv[1:])
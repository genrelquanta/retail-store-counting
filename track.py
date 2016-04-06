import numpy as np
import scipy
import sys, getopt

from helper import *
from simpleTrack import *
from kalmanFilterTrack import *




def main(argv):
   try:
      opts, args = getopt.getopt(argv,"hvp:f:t:o:",["path=","file=","tracker=","output="])
   except getopt.GetoptError:
      print 'track.py -p <path-to-data> -t <tracker-to-use> -v <visualize?> -o <output-directory>'
      sys.exit(2)

   tracker = 'simple'
   vis = False
   for opt, arg in opts:
      if opt == '-h':
         print 'track.py -p <path-to-data> -f <data-file> -t <tracker-to-use> -v <visualize?> -o <output-directory>'
         sys.exit()
      elif opt in ("-p", "--path"):
         path = arg
      elif opt in ("-f", "--file"):
         filename = arg
      elif opt in ("-t", "--tracker"):
         tracker = arg
      elif opt == '-v':
         vis = True
      elif (opt in ("-o", "--output")) and vis:
         savDir = arg
   detectSeq = readData(path+filename)
   if tracker=='kalman':
      objTrack = trackObjKalman(detectSeq, 1e3, 0.5, 10, 150, 0.8, 10, path)
   elif tracker =='simple':
      objTrack = trackObj(detectSeq, 200, 20, path)
   stats = countStats(objTrack, 400)
   print 'Number of people entering store = ',stats[0]
   print 'Number of people exiting store = ',stats[1]
   if vis:
      visualize(detectSeq,objTrack,path,savDir,bbox=False,patch=True)

if __name__ == "__main__":
   main(sys.argv[1:])










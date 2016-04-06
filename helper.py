import re
import numpy as np
import scipy
from scipy import misc
import time
import colorsys
import os

"""
This file implements helper functions used for reading, manipulating
and visualizing data
"""


def readData(filename):
  """
  Reads and parses the input data file.
  
  Input: Name of file to be parsed
  
  Output: A list of pairs of the form (img,detections),
  where img is the name of the image file and detections is a list 
  of tuples of the form (num_1,num_2,num_3,num_4), where num_i are 
  the co-ordinates of the detections.
  """
  fileRegEx = re.compile('\w+\.jpg')
  dataRegEx = re.compile('(\d+\.\d+),\s(\d+\.\d+),\s(\d+\.\d+),\s(\d+\.\d+)')
  output = []
  f = open(filename,'r')
  for line in f:
    detections = list()
    img = re.findall(fileRegEx,line)[0]
    dataS = re.findall(dataRegEx,line)
    for tup in dataS:
      detections.append((float(tup[0]),float(tup[1]),float(tup[2]),float(tup[3])))
    output.append((img,detections))
  return output

def featureExtract(img,patches,centr=True,patchFeat=False):
  """
  Reads image file img and extracts features (centroids only in this
  implementation), can be extended to include other appearance based
  features
  
  Input: Image filename and patch (a list of patches)
  
  Output: A list of feature vectors
  """
  #image = float(misc.imread(img))
  if (not patchFeat and centr):
    feats = np.zeros((2,len(patches)))
    for patch in range(len(patches)):
      feats[0,patch] = (patches[patch][0]+patches[patch][2])/2
      feats[1,patch] = (patches[patch][1]+patches[patch][3])/2

  return feats

def visualize(detectSeq,objTrack,dir,savDir,bbox=True,patch=False):

  """
  Generates sequence of images with tracked objects marked by bounding 
  boxes or colored patches
  
  Input: 
  -detectSeq : List of detections read using readData helper function
  -objTrack : Object tracks generated using object tracker function
  -dir : directory which stores the input image sequence
  -savDir : directory to which the output image sequence is written
  -bbox : boolean variable indicating if tracked objects should be marked 
  with bounding boxes
  -patch : boolean variable indicating if tracked objects should be marked
  with colored patches
  
  Output: Writes the marked image sequence to savDir . Use 
  ffmpeg -f image2 -r 12 -i img%d.jpg -vcodec mpeg4 -y movie.mp4
  to convert the sequence of images to video
  """
  imgSeq = [detectSeq[i][0] for i in range(len(detectSeq))]
  frameToCords = {}
  trackID = 0
  for obj in objTrack:
    for track in obj:
      if track[1] in frameToCords:
        frameToCords[track[1]].append((track[0],trackID))
      else:
        frameToCords[track[1]] = [(track[0],trackID)]
    trackID += 1
  RGB = genCols(len(objTrack))
  
  print "Generating tracking video..."
  for im in range(len(imgSeq)):
    img = misc.imread(dir+imgSeq[im])
    for obj in frameToCords[im]:
      for col in range(0,3):
        if bbox:
          img[int(obj[0][1]):int(obj[0][1])+3,int(obj[0][0]):int(obj[0][2]),col] = RGB[obj[1]][col]*255
          img[int(obj[0][3])-3:int(obj[0][3]),int(obj[0][0]):int(obj[0][2]),col] = RGB[obj[1]][col]*255
          img[int(obj[0][1]):int(obj[0][3]),int(obj[0][0]):int(obj[0][0])+3,col] = RGB[obj[1]][col]*255
          img[int(obj[0][1]):int(obj[0][3]),int(obj[0][2])-3:int(obj[0][2]),col] = RGB[obj[1]][col]*255
        elif patch:
          img[int(obj[0][1]):int(obj[0][3]),int(obj[0][0]):int(obj[0][2]),col] = RGB[obj[1]][col]*255
    misc.imsave(savDir+'img{0}.jpg'.format(im+1), img)
  print "Saving video..."
  os.chdir(savDir)
  os.system('ffmpeg -f image2 -r 12 -i img%d.jpg -vcodec mpeg4 -y movie.mp4')


def genCols(N):
    
  """
  Generates N distinct colors
  
  Input: Number of colors to be generated
  
  Output: List of RGB tuples of colors generated
  """
  HSV = [(c*1.0/N, 0.5, 0.5) for c in range(N)]
  RGB = map(lambda x: colorsys.hsv_to_rgb(*x),HSV)
  return RGB

def countStats(objTrack, crossNum):
    
  """
  Uses object tracks to count the number of people entering and exiting the store
  
  Input: 
  -objTrack : Tracks of objects
  -crossNum : X-co-ordinate to use for counting crossings
  
  Output: Number of people entering and exiting the store
  """
  numEntry = 0
  numExit = 0
  for obj in objTrack:
    minX = min([obj[i][0][1] for i in range(len(obj))])
    maxX = max([obj[i][0][3] for i in range(len(obj))])
    dir = (obj[0][0][1]-obj[len(obj)-1][0][1])<0
    toCount = crossNum>=minX and crossNum<maxX
    if dir and toCount:
      numEntry += 1
    if (not dir) and toCount:
      numExit += 1

  return (numEntry,numExit)




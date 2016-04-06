import numpy as np
import scipy

from scipy import misc
from helper import *

"""
This file implements tracking using correspondence between
neighboring images based on object motion. The objects are
assumed to be moving with small constant velocities. Correspondence
is computed using greedy assignment algorithm. 
"""

def matchError(patches_1, patches_2, thresh):
  """
  Computes error between patches in patches_1 and patches_2
  and assigns patches in patches_2 to patches in patches_1.
  Uses a greedy algorithm which greedily matches the minimum
  cost patch pairs so far.
  
  Input: Two arrays of patch features
  
  Output: A list indicating assignment of each patch in patches_2
  to a patch in patches_1, -1 if no assignment
  """
  len_1 = patches_1.shape[1]
  len_2 = patches_2.shape[1]
  
  match = -1*np.ones(len_2)
  p_1,p_2 = np.meshgrid(np.arange(0,len_1),np.arange(0,len_2))
  p_1x = patches_1[0,p_1]
  p_1y = patches_1[1,p_1]
  p_2x = patches_2[0,p_2]
  p_2y = patches_2[1,p_2]
  
  # Compute cost of matching

  error = np.sqrt((p_1x-p_2x)**2 + (p_1y-p_2y)**2)
  error[error>thresh] = np.inf
  
  # Greedy Assignment

  while np.min(error)<np.inf:
    p_2m, p_1m = np.where(error==np.min(error))
    match[p_2m[0]] = p_1m[0]
    error[p_2m[0],0:len_1] = np.inf
    error[0:len_2,p_1m[0]] = np.inf

  return match

def trackObj(detectSeq, thresh, trackThresh, dir):
  """
  Tracks objects across frames based on constant velocity assumption
  and greedy matching
  
  Input: 
  -detectSeq : List of detections read using readData helper function
  -thresh : threshold to use for matching
  -trackThresh : threshold for length of tracks
  -dir : directory where input images reside
  
  Output: A list of tracks of objects (objects are represented by 
  the same four numbers used for representing detections)
  """
  # Initialize new tracks from the first frame
  
  objID = np.arange(len(detectSeq[0][1]))
  objVel = np.zeros((2,len(objID)))
  feats = featureExtract(dir+detectSeq[0][0],detectSeq[0][1],centr=True,patchFeat=False)

  objTrack = list()
  for i in range(len(objID)):
    objTrack.append([(detectSeq[0][1][i],0)])
  
  for frame in range(0,len(detectSeq)-1):
     
    # Predict position of past detections assuming constant velocity
    
    if objVel.shape[0]>0:
      patches_1 = feats+objVel
    else:
      patches_1 = np.zeros(0)


    # Compute features of new detections
    
    patches_2 = featureExtract(dir+detectSeq[frame+1][0],detectSeq[frame+1][1],centr=True,patchFeat=False)
    newObjID = np.zeros(len(detectSeq[frame+1][1]))
    newObjVel = np.zeros((2,len(newObjID)))

    # Assign new detections to existing tracks

    if(patches_1.shape[0]>0 and patches_2.shape[0]>0):
      match = matchError(patches_1, patches_2, thresh)
    else:
      match = -1*np.ones(patches_2.shape[1])

    # Update the existing tracks and start new tracks for unassigned detections

    for j in range(patches_2.shape[1]):
      if match[j]==-1:
        newObjID[j] = len(objTrack)
        objTrack.append([(detectSeq[frame+1][1][j],frame+1)])
      else:
        newObjID[j] = objID[match[j]]
        lastObjInd = len(objTrack[int(newObjID[j])])-1
        newObjVel[0,j] = patches_2[0,j]-objTrack[int(newObjID[j])][lastObjInd][0][0]
        newObjVel[1,j] = patches_2[1,j]-objTrack[int(newObjID[j])][lastObjInd][0][1]
        objTrack[int(newObjID[j])].append((detectSeq[frame+1][1][j],frame+1))
    objID = newObjID
    objVel = newObjVel
    feats = patches_2
        
  # Only keep tracks which are long enough

  filteredTracks = []
  for track in objTrack:
    if len(track)>trackThresh:
      filteredTracks.append(track)

  return filteredTracks

  



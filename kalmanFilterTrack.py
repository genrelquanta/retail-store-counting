import numpy as np
import scipy

from scipy import misc
from numpy.linalg import inv
from munkres import Munkres
from helper import *

"""
This file implements kalman filter and tracking using kalman filter.
The implementation of kalman filter based tracking is based on a
a MATLAB tutorial.
"""


class kalmanFilter(object):
  """
  kalman filter used for tracking objects in video. The object state 
  is defined by [x,y,dx,dy] where x and y are locations of centroid
  of the detection and dx and dy are the motion vectors or velocities.
  The measurement variable is [x,y], the locations of centroids of the
  detections.
  
  Parameters:
  - x : Object state
  - A : Transition Matrix
  - H : Measurement Matrix
  - Q : Process Noise Matrix
  - R : Measurement Noise Matrix
  - P : Covariance matrix corresponsing to process estimation
  """
  
  def __init__(self, x_init, pVar, qVar, rVar):
    """
    Initializes a new kalman filter.
    
    Inputs:
    - x_init : Initial state of the filter
    - pVar : Initial variance of the process estimate covariance matrix (initialized
    as a I*pvar matrix where I is identity matrix
    - qVar : scale factor for the process noise matrix
    - rvar : measurement noise variance
    """
    self.x = x_init
    
    self.A = np.eye(4,4)
    self.A[0,2] = 1
    self.A[1,3] = 1
    
    self.H = np.zeros((2,4))
    self.H[0,0] = 1
    self.H[1,1] = 1
    
    self.Q = np.array([[1/4,0,1/2,0],[0,1/4,0,1/2],[1/2,0,1,0],[0,1/2,0,1]])*qVar
    
    self.P = np.eye(4,4)*pVar
    
    self.R = np.eye(2,2)*rVar
  
  def kalmanPredict(self):
    """
    Predicts the next location of the object
    """
      
    self.x = np.dot(self.A,self.x.T)
    self.P = np.dot(self.A,np.dot(self.P,(self.A).T)) + self.Q


  def kalmanUpdate(self,z):
    """
    Updates the new location of the object based on new measurement
    
    - z : New measured position of the object
    """
    
    y = z - np.dot(self.H,self.x.T)
    S = np.dot(self.H,np.dot(self.P,(self.H).T)) + self.R
    K = np.dot(self.P,np.dot((self.H).T, inv(S)))
    self.x = self.x + np.dot(K,y)
    self.P = np.dot((np.eye((self.x).shape[0],(self.x).shape[0]) - np.dot(K,self.H)),self.P)


def featureExtract(img,patches,centr=True,patchFeat=False):
  """
  Reads image file img and extracts features
  
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
               
def matchErrorKalman(objTrack, newDetect, noCost):
  """
  Assigns new detections to existing tracks by minimizing the total cost
  of assignment. Assignment is done using maximum weight matching on 
  a bipartite graph of tracks and new detections
  
  Input: 
  - objTrack : Existing tracks of objects
  - newDetect : New detections to be assigned
  - noCost: Cost of not assigning to any track. If an object moves too far,
  a new track is started. Controls the segmentation of the tracks
  
  Output: A list indicating which patch each patch in patches_2
  corresponds to, -1 if no match
  """
  # Compute cost of assignment
  cost = []
  trackID = objTrack.keys()
  for track in trackID:
    costRow = list()
    for det in range(newDetect.shape[1]):
      costRow.append(np.sqrt((objTrack[track]['kalman'].x[0]-newDetect[0,det])**2 + (objTrack[track]['kalman'].x[1]-newDetect[1,det])**2))
    cost.append(costRow)
  
  # Use hungarian algorithm for maximum matching in weighted bipartite graph
  m = Munkres()
  match = -1*np.ones(newDetect.shape[1])
  assign = m.compute(cost)
  
  for row, column in assign:
    if cost[row][column]>noCost:
      match[column]=-1
    else:
      match[column]=trackID[row]
  
  return match
  

def trackObjKalman(detectSeq, pVar, qVar, rVar, noCost, visTh, ageTh, dir):
  """
  Tracks objects across frames 
  
  Input: 
  -detectSeq : List of detections read using readData helper function
  -pVar, qVar, rVar : Parameters for kalman filter
  -noCost : Parameter for matching algorithm
  -visTh : Threshold to indicate for what fraction of it's life, a track
  should be visible
  -ageTh : threshold to remove short lived tracks
  -dir : directory where input images reside
  
  Output: A list of tracks of objects (objects are represented by 
  the same four numbers used for representing detections)
  """
  
  # Initialize object tracks
  feats = featureExtract(dir+detectSeq[0][0],detectSeq[0][1],centr=True,patchFeat=False)
  objTrack = dict()
  for i in range(len(detectSeq[0][1])):
    objTrack[i] = {'loc':[(detectSeq[0][1][i],0)],'kalman': kalmanFilter(np.array([feats[0,i],feats[1,i],0,0]),pVar,qVar,rVar), 'age': 1, 'vis' : 1, 'inv' : 0}
  nextID = len(objTrack)
  
  for frame in range(0,len(detectSeq)-1):
               
    # Predict new locations for existing tracks
    for track in objTrack:
      objTrack[track]['kalman'].kalmanPredict()
               
    # Assign new detections to existing trackss
    newDetect = featureExtract(dir+detectSeq[frame+1][0],detectSeq[frame+1][1],centr=True,patchFeat=False)
    match = matchErrorKalman(objTrack, newDetect, noCost)
               
    # Update tracks to which new detections have been assigned
    
    for j in range(len(match)):
      if match[j] !=-1:
        objTrack[match[j]]['kalman'].kalmanUpdate(newDetect[0:2,j])
        objTrack[match[j]]['age'] += 1
        objTrack[match[j]]['vis'] += 1
        objTrack[match[j]]['inv'] = 0
        lastPos = len(objTrack[match[j]]['loc'])-1
        lastVal = objTrack[match[j]]['loc'][lastPos][0]
        dx = objTrack[match[j]]['kalman'].x[0]-(lastVal[0]+lastVal[2])/2
        dy = objTrack[match[j]]['kalman'].x[1]-(lastVal[1]+lastVal[3])/2
        objTrack[match[j]]['loc'].append(((lastVal[0]+dx,lastVal[1]+dy,lastVal[2]+dx,lastVal[3]+dy),frame+1))
               
    # Update tracks which haven't been assigned

    for track in objTrack:
      if track not in match:
        objTrack[track]['age'] += 1
        objTrack[track]['inv'] += 1
        lastPos = len(objTrack[track]['loc'])-1
        lastVal = objTrack[track]['loc'][lastPos][0]
        dx = objTrack[track]['kalman'].x[0]-(lastVal[0]+lastVal[2])/2
        dy = objTrack[track]['kalman'].x[1]-(lastVal[1]+lastVal[3])/2
        objTrack[track]['loc'].append(((lastVal[0]+dx,lastVal[1]+dy,lastVal[2]+dx,lastVal[3]+dy),frame+1))
    
    # Delete tracks which are visible for a short period of time
    toDel = []
    for track in objTrack:
      visFrac = float(objTrack[track]['vis'])/float(objTrack[track]['age'])
      if (objTrack[track]['age']<ageTh and visFrac < visTh):
        toDel.append(track)
    for track in toDel:
      del objTrack[track]
               
    # Start new tracks at unassigned detections
    
    for j in range(newDetect.shape[1]):
      if match[j] == -1:
        objTrack[nextID] = {'loc':[(detectSeq[frame+1][1][j],frame+1)],'kalman': kalmanFilter(np.array([newDetect[0,j],newDetect[1,j],0,0]),pVar,qVar,rVar), 'age': 1, 'vis' : 1, 'inv' : 0}
        nextID += 1
   
   # Final tracks of objects
  finalTrack = []
  for track in objTrack:
    finalTrack.append(objTrack[track]['loc'])
  
  return finalTrack

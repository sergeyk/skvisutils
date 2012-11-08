"""
Miscellaneous utilities.
"""
import numpy as np

from skpyutils import Table

def nms_detections_table(dets_table, overlap=0.5):
  return Table(nms_detections(dets_table.arr, dets_table.cols), dets_table.cols)

def nms_detections(dets, cols, overlap=0.5):
  """
  Perform non-maximum suppression (NMS) on the given detections.

  NMS greedily selects high-scoring detections and skips detections that are
  significantly covered by a previously selected detection.

  This version is translated from Matlab code_ by Tomasz Maliseiwicz,
  who sped up Pedro Felzenszwalb's voc-release version.

  .. _code: https://gist.github.com/1144423/bd06b536bd991e1b8e0991b811b999b1b83b9c86)

  Args:
    dets (NxD ndarray): every row is a detection.
    
    cols (list of len D): column names for the detections.
      Must include 'x', 'y', 'w', 'h', 'score'

    overlap (float): [0.5 default] min overlap ratio.
  
  Returns:
    dets (NxD ndarray): detections that remain after the suppression.

  Raises:
    none
  """
  if np.shape(dets)[0] < 1:
    return dets 

  x1 = dets[:,cols.index('x')]
  y1 = dets[:,cols.index('y')]

  w = dets[:,cols.index('w')]
  h = dets[:,cols.index('h')]
  x2 = w+x1-1
  y2 = h+y1-1 
  s = dets[:,cols.index('score')]

  area = w*h
  ind = np.argsort(s)

  pick = [] 
  counter = 0
  while len(ind)>0:
    last = len(ind)-1
    i = ind[last] 
    pick.append(i)
    counter += 1
    
    xx1 = np.maximum(x1[i], x1[ind[:last]])
    yy1 = np.maximum(y1[i], y1[ind[:last]])
    xx2 = np.minimum(x2[i], x2[ind[:last]])
    yy2 = np.minimum(y2[i], y2[ind[:last]])
    
    w = np.maximum(0., xx2-xx1+1)
    h = np.maximum(0., yy2-yy1+1)
    
    o = w*h / area[ind[:last]]
    
    to_delete = np.concatenate((np.nonzero(o>overlap)[0],np.array([last])))
    ind = np.delete(ind,to_delete)
  return dets[pick,:]

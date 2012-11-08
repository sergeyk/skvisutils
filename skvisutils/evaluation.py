"""
Methods for evaluating detection and classification performance over a dataset.
Includes methods to plot results and output a single HTML dashboard.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mako.template import Template

from skpyutils import mpi, TicToc, Table

from skvisutils import Dataset, BoundingBox

EVAL_TEMPLATE_FILENAME = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'support/dashboard_template.html'))

def evaluate_per_class_detections(dataset, dets, dirname, force=False):
  """
  Output evaluations of detections over the dataset in per-class regime.

  Will distribute per-class evaluation computation over MPI if available.

  Args:
    dataset (skvisutils.Dataset): the dataset of the detections

    dets (skpyutils.Table): the detections

    dirname (string): path to directory where the evaluations will be output

    force (bool): if False, will not re-compute PR curves if files are there

  Returns:
    aps (list of floats), and outputs plots and supporting files to dirname
  """
  num_classes = len(dataset.classes)
  dist_aps = np.zeros(num_classes)
  for cls_ind in range(mpi.comm_rank, num_classes, mpi.comm_size):
    object_class = dataset.classes[cls_ind] 
    cls_dets = dets.filter_on_column('cls_ind',cls_ind)
    cls_gt = dataset.get_det_gt_for_class(object_class, with_diff=True)
    ap = compute_and_plot_pr(cls_dets, cls_gt, object_class, dirname, force)
    dist_aps[cls_ind] = ap
  aps = np.zeros(num_classes)
  mpi.safebarrier()
  mpi.comm.Reduce(dist_aps,aps)
  return aps

def evaluate_multiclass_detections(dataset, dets, dirname, force=False):
  """
  Output evaluations of detections over the dataset in the multi-class regime.

  Args:
    dataset (skvisutils.Dataset): the dataset of the detections

    dets (skpyutils.Table): the detections

    dirname (string): path to directory where the evaluations will be output

    force (bool): if False, will not re-compute PR curves if files are there

  Returns:
    ap (float), and outputs plots and supporting files to dirname
  """
  gt = dataset.get_det_gt(with_diff=True)
  # TODO filename = opjoin(self.results_path, 'pr_whole_multiclass')
  if force or not opexists(filename):
    print("Evaluating %d dets in the multiclass setting..."%dets.shape[0])
    ap = compute_and_plot_pr(dets, gt, 'multiclass', dirname)
  return ap

def evaluate_detections_whole(dataset, dets, dirname, force=False):
  """
  Output evaluations of detections over the whole dataset in multi-class and
  per-class regimes.

  Will distribute per-class evaluation computation over MPI if available.

  Args:
    dataset (skvisutils.Dataset): the dataset of the detections

    dets (skpyutils.Table): the detections

    dirname (string): path to directory where the evaluations will be output

    force (bool): if False, will not re-compute PR curves if files are there

  Returns:
    none, but outputs plot files and a single HTML dashboard.
  """
  aps = evaluate_per_class_detections(dataset, dets, dirname, force)

  if mpi.comm_rank == 0:
    # TODO: re-run per-class to make sure they plot?

    ap_mc = evaluate_multiclass_detections(dataset, dets, dirname, force)

    # Write out the information to a single overview file
    filename = os.path.join(dirname, 'aps_whole.txt')
    with open(filename, 'w') as f:
      f.write("Multiclass: %.3f\n"%ap_mc)
      f.write(','.join(dataset.classes)+'\n')
      f.write(','.join(['%.3f'%ap for ap in aps])+'\n')

    # Assemble everything in one HTML file, the Dashboard
    names = list(self.dataset.classes)+['mean ap','multiclass']
    aps = aps.tolist()+[np.mean(aps), ap_mc]
    
    template = Template(EVAL_TEMPLATE_FILENAME)
    filename = os.path.join(dirname,'whole_dashboard.html')
    with open(filename, 'w') as f:
      f.write(template.render(names=names, aps=aps))
  mpi.safebarrier()

def compute_and_plot_pr(dets, gt, name, dirname, force=False):
  """
  Compute the precision-recall curve from the given detections and ground truth
  and output the vectors and plots to files.

  Args:
    dets (skpyutils.Table): detections on the dataset

    gt (skpyutils.Table): dataset ground truth

    name (string): name for the curve, displayed in the plot legend

    dirname (string): path to dirname where files will be output

    force (bool): if False, will not recompute curves or plot if data is there

  Returns:
    ap (float), and outputs files to dirname
  """
  # TODO: isn't there some problem with MPI-distributed jobs not plotting?
  filename = os.path.join(dirname, 'pr_whole_{0}.txt'.format(name))
  if force or not opexists(filename):
    [ap,rec,prec] = self.compute_det_pr(dets, gt)
    try:
      plot_filename = os.path.join(dirname, 'pr_whole_{0}.png'.format(name))
      plot_pr(ap,rec,prec,name,plot_filename)
    except:
      pass
    with open(filename, 'w') as f:
      f.write("%f\n"%ap)
      for i in range(np.shape(rec)[0]):
        f.write("%f %f\n"%(rec[i], prec[i]))
  else:
    with open(filename) as f:
      ap = float(f.readline())
  return ap

def plot_pr(ap, rec, prec, name, filename, force=False):
  """
  Plot the Precision-Recall curve, saving the png to filename.

  Args:
    ap (float): average precision

    rec (ndarray): recall values

    prec (ndarray): precision values

    name (string): name to use in the legend

    filename (string): path to the resulting saved png

  Returns:
    none
  """
  # TODO: make return Figure
  if opexists(filename) and not force:
    print("plot_pr: not doing anything as file exists")
    return
  label = "%s: %.3f"%(name,ap)
  plt.clf()
  plt.plot(rec,prec,label=label,color='black',linewidth=5)
  plt.xlim(0,1)
  plt.ylim(0,1)
  plt.legend(loc='lower left')
  plt.xlabel('Recall',size=16)
  plt.ylabel('Precision',size=16)
  plt.grid(True)
  plt.savefig(filename)

def compute_det_map(dets, gt, values=None, det_perspective=False, min_overlap=0.5):
  """
  Compute Mean Average Precision of the detections for the given ground truth.
  Evaluates each class separately.

  Args:
    dets (skpyutils.Table): must have relevant columns

    gt (skpyutils.Table): must have relevant columns

    values (dict): cls_ind (in the dataset) -> value lookup.
      If None, initialize to all 1's (resulting in unweighted mean).

    det_perspective (bool):
      if True, only classes present in dets are evaluated.
      if False, only classes present in the ground truth are evaluated.

  Raises:
    none
  """
  if det_perspective:
    unique_cls_inds = np.unique(dets.subset_arr('cls_ind'))
  else:
    unique_cls_inds = np.unique(gt.subset_arr('cls_ind'))

  if values is None:
    values = dict( [(cls_ind, 1.) for cls_ind in unique_cls_inds] )
  values_sum = 1.*np.sum([values[cls_ind] for cls_ind in unique_cls_inds])

  aps = []
  for cls_ind in unique_cls_inds:
    d = dets.filter_on_column('cls_ind',cls_ind)
    g = gt.filter_on_column('cls_ind',cls_ind)
    ap,_,_ = compute_det_pr(d, g, min_overlap)
    aps.append(ap*values[cls_ind])
  return np.sum(aps/values_sum)

def compute_det_pr(dets, gt, min_overlap=0.5):
  return compute_det_pr_and_hard_neg(dets, gt, min_overlap)[:3]

def compute_det_hard_neg(dets, gt, min_overlap=0.5):
  return compute_det_pr_and_hard_neg(dets, gt, min_overlap)[3]
  
def compute_det_pr_and_hard_neg(dets, gt, min_overlap=0.5):
  """
  Compute the Precision-Recall and find hard negatives of the given detections
  for the ground truth.

  Args:
    dets (skpyutils.Table): detections

    gt (skpyutils.Table): detectin ground truth
      Can be for a single image or a whole dataset, and can contain either all
      classes or a single class. The 'cls_ind' column must be present in
      either case.

      Note that depending on these choices, the meaning of the PR evaluation
      is different. In particular, if gt is for a single class but detections
      are for multiple classes, there will be a lot of false positives!

    min_overlap (float): minimum required area of union of area of
      intersection overlap for a true positive.

  Returns:
    (ap, recall, precision, hard_negatives): tuple of
      (float, list, list, list, sorted_dets), where the lists are 0/1 masks
      onto the sorted dets.
  """
  tt = TicToc().tic()

  # if dets or gt are empty, return 0's
  nd = dets.arr.shape[0]
  if nd < 1 or gt.shape[0] < 1:
    ap = 0
    rec = np.array([0])
    prec = np.array([0])
    hard_negs = np.array([0])
    return (ap,rec,prec,hard_negs)
  
  # augment gt with a column keeping track of matches
  cols = list(gt.cols) + ['matched']
  arr = np.zeros((gt.arr.shape[0],gt.arr.shape[1]+1))
  arr[:,:-1] = gt.arr.copy()
  gt = Table(arr,cols)

  # sort detections by confidence
  dets = dets.copy()
  dets.sort_by_column('score',descending=True)

  # match detections to ground truth objects
  npos = gt.filter_on_column('diff',0).shape[0]
  tp = np.zeros(nd)
  fp = np.zeros(nd)
  hard_neg = np.zeros(nd)
  for d in range(nd):
    if tt.qtoc() > 15:
      print("... on %d/%d dets"%(d,nd))
      tt.tic()

    det = dets.arr[d,:]

    # find ground truth for this image
    if 'img_ind' in gt.cols:
      img_ind = det[dets.ind('img_ind')]
      inds = gt.arr[:,gt.ind('img_ind')] == img_ind
      gt_for_image = gt.arr[inds,:]
    else:
      gt_for_image = gt.arr
    
    if gt_for_image.shape[0] < 1:
      # false positive due to a det in image that does not contain the class
      # NOTE: this can happen if we're passing ground truth for a class
      fp[d] = 1 
      hard_neg[d] = 1
      continue

    # find the maximally overlapping ground truth element for this detection
    overlaps = BoundingBox.get_overlap(gt_for_image[:,:4],det[:4])
    jmax = overlaps.argmax()
    ovmax = overlaps[jmax]

    # assign detection as true positive/don't care/false positive
    if ovmax >= min_overlap:
      if gt_for_image[jmax,gt.ind('diff')]:
        # not a false positive because object is difficult
        None
      else:
        if gt_for_image[jmax,gt.ind('matched')] == 0:
          if gt_for_image[jmax,gt.ind('cls_ind')] == det[dets.ind('cls_ind')]:
            # true positive
            tp[d] = 1
            gt_for_image[jmax,gt.ind('matched')] = 1
          else:
            # false positive due to wrong class
            fp[d] = 1
            hard_neg[d] = 1
        else:
          # false positive due to multiple detection
          # this is still a correct answer, so not a hard negative
          fp[d] = 1
    else:
      # false positive due to not matching any ground truth object
      fp[d] = 1
      hard_neg[d] = 1
    # NOTE: must do this for gt.arr to get the changes we made to gt_for_image
    if 'img_ind' in gt.cols:
      gt.arr[inds,:] = gt_for_image

  ap,rec,prec = compute_rec_prec_ap(tp,fp,npos)
  return (ap,rec,prec,hard_neg)

def compute_cls_map(clses,gt):
  """
  Compute mean classification AP.

  The results should be for multiple classes, and we average per-class APs.
  There should only be one instance of each img_ind in clses.

  Args:
    clses (skpyutils.Table): classifications

    gt (skpyutils.Table): ground truth
        Should be of the format in Dataset.get_cls_ground_truth().
        NOTE:  must be for the whole dataset!

  Returns:
    map (float)
  """
  # TODO: incorporate values
  if not clses.shape[0]>0:
    return 0
  if 'img_ind' in clses.cols:
    img_inds = clses.subset_arr('img_ind')
    assert(len(img_inds)==len(np.unique(np.array(img_inds))))
    gt = gt.row_subset(img_inds)
  aps = []
  for col in gt.cols:
    ap,rec,prec=cls.compute_cls_pr(clses.subset_arr(col),gt.subset_arr(col))
    aps.append(ap)
  return np.mean(aps)

def compute_cls_pr(confidences,gt):
  """
  Compute classification Precision, Recall, and AP.

  Args:
    confidences ((Nx1) ndarray): classification confidences

    gt ((Nx1) ndarray): ground truth
  
  Returns:
    (ap,recall,precision): (float, ndarray, ndarray)
  """
  ind = np.argsort(-confidences, axis=0)
  tp = gt[ind]==True
  fp = gt[ind]==False
  npos = np.sum(gt==True)
  return cls.compute_rec_prec_ap(tp,fp,npos)

def compute_rec_prec_ap(tp,fp,npos):
  """
  Compute the Recall and Precision vectors and the area under the PR curve.

  Args:
    tp ((Nx1) ndarray): binary vector of whether the detection was a tp
    
    fp ((Nx1) ndarray): analogous to tp

    npos (int): number of positives possible

  Returns:
    (ap, rec, prec): (float, (Nx1) ndarray, (Nx1) ndarray)
  """
  fp=np.cumsum(fp)
  tp=np.cumsum(tp)
  if npos==0:
    rec = np.zeros(tp.shape)
  else:
    rec=1.*tp/npos
  prec=1.*tp/(fp+tp)
  prec = np.nan_to_num(prec)
  ap = compute_ap(rec,prec)
  return (ap,rec,prec)

def compute_ap(rec,prec):
  """
  Compute piecewise area under the curve of the recall and precision vectors.

  Args:
    rec ((Nx1) ndarray): recall values
    
    prec ((Nx1) ndarray): precision values

  Returns:
    ap (float)
  """
  assert(np.all(rec>=0) and np.all(rec<=1))
  assert(np.all(prec>=0) and np.all(prec<=1))
  mprec = np.hstack((0,prec,0))
  mrec = np.hstack((0,rec,1))

  # make sure prec is monotonically decreasing
  for i in range(len(mprec)-1,0,-1):
    mprec[i-1]=max(mprec[i-1],mprec[i])

  # find piecewise area under the curve
  i = np.add(np.nonzero(mrec[1:] != mrec[0:-1]),1)
  ap = np.sum((mrec[i]-mrec[np.subtract(i,1)])*mprec[i])
  return ap

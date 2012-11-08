from sklearn.cross_validation import KFold
import cPickle
import os
import re
import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
from skpyutils import TicToc, Table, skutil

from skvisutils import Image, SlidingWindows, Config

class Dataset(object):
  """
  Representation of an object recognition dataset.

  Provides methods to load or construct from different sources of data,
  to get classification and detection ground truth,
  and to construct sets of train/test data.

  Initialize the class with a Config and an optional name,
  but the actual loading of data must be done in a follow-up call:
  load_from_pascal, load_from_json, or generate_synthetic.
  """

  def __init__(self, config, name=None):
    """
    Initialize with Config class, which determines where to load PASCAL from
    and where to cache the loaded dataset and to output figures if needed.

    Args:
      config (skvisutils.config.Config): sets paths and such

      name (string): [optional] identifying information
    
    Returns:
      self

    Raises:
      none
    """
    assert(isinstance(config,Config))
    self.config = config
    self.name = name

    self.classes = []
    self.values = []
    self.images = []
    self.current_fold = -1
    self.cached_det_ground_truth = {}

  def after_load(self):
    """
    Make sure that image names are unique, and set class values to uniform.

    Returns:
      self
    """
    self.image_names = [image.name for image in self.images]
    assert(len(self.image_names)==len(np.unique(self.image_names)))
    self.set_class_values('uniform')
    return self

  def num_classes(self):
    return len(self.classes)

  def get_ind(self,cls):
    return self.classes.index(cls)

  def num_images(self):
    return len(self.images)

  def get_img_ind(self,image):
    return self.images.index(image)

  def __repr__(self):
    s = "Dataset:\n"
    s += "Name: {0}\n".format(self.name)
    s += "Num images: {0}\n".format(len(self.images))
    s += "Classes: {0}\n".format(self.classes)
    return s

  def get_name(self):
    """
    Return name with number of images appended.
    """
    return "%s_%s"%(self.name,self.num_images())

  def set_class_values(self,mode='uniform'):
    """
    Set class values to be uniform or inversely proportional to priors.

    Args:
      mode (string): in ['uniform','inverse_prior']

    Returns:
      None

    Raises:
      ValueError
    """
    if mode=='uniform':
      self.values = 1.*np.ones(len(self.classes))/len(self.classes)
    elif mode=='inverse_prior':
      gt = self.get_cls_ground_truth(with_diff=False,with_trun=True)
      prior = 1.*gt.sum(0)/gt.shape[0]
      self.values = 1./prior
      self.values /= np.sum(self.values)
    else:
      raise ValueError("Unknown mode")

  def generate_synthetic(self):
    """
    Generate a synthetic dataset of 4 classes that follows some simple
    but strong cooccurence rules.

    Returns:
      self
    """
    num_images = 1000
    choice_probs = {
      (1,0,0,0):2,
      (0,1,0,0):1,
      (0,0,1,0):1,
      (0,0,0,1):1,
      (1,1,0,0):2,
      (1,0,1,0):0,
      (1,0,0,1):0,
      (0,1,1,0):0,
      (0,1,0,1):1,
      (0,0,1,1):2,
      (1,1,1,0):0,
      (1,1,0,1):1,
      (1,0,1,1):1,
      (0,1,1,1):2}
    probs = np.array(choice_probs.values())
    cum_probs = np.cumsum(1.*probs/np.sum(probs))    
    self.classes = ['A','B','C','D']
    for i in range(0,num_images):
      image = Image(100,100,self.classes,str(i))
      choice = np.where(cum_probs>np.random.rand())[0][0]
      objects = []
      for cls_ind,clas in enumerate(choice_probs.keys()[choice]):
        if clas == 1:
          objects.append(np.array([0,0,0,0,cls_ind,0,0]))
      image.objects_table = Table(np.array(objects), Image.columns)
      self.images.append(image)

    return self.after_load()

  def load_from_json(self, filename):
    """
    Load all parameters of the dataset from a JSON file.

    Args:
      filename (string): path to JSON data file

    Returns:
      self
    """
    with open(filename) as f: json_data = json.load(f)
    self.classes = json_data['classes']
    self.images = [Image.load_from_json(self.classes,image) for image in json_data['images']]
    return self.after_load()

  def load_from_pascal(self, name, force=False):
    """
    Load PASCAL VOC data, loading from cache if it exists at a canonical path.
    Both the PASCAL and cache paths should be set in self.config.
    After loading, saves the images and classes to cache.

    Uses Image class to load data from the PASCAL XML files.

    Args:
      name (string): in ['train','trainval','val','test']

      force (bool): [optional] if True, do not load from cache
    
    Returns:
      self

    Raises:
      self
    """
    tt = TicToc().tic()
    print("Dataset: %s"%self.get_name())

    filename = self.config.get_cached_dataset_filename(self.get_name())
    if os.path.exists(filename) and not force:
      with open(filename) as f:
        cached = cPickle.load(f)
        self.classes = cached['classes']
        self.images = cached['images']
        print("...loaded from cache in %.2f s"%tt.qtoc())
    
    else:
      print("...loading from scratch")
      self.classes = self.config.pascal_classes 
      
      with open(self.config.get_pascal_path(name)) as f:
        imgset = [line.strip() for line in f.readlines() if len(line)>0]
      
      images_dir = os.path.join(self.config.pascal_dir, "JPEGImages")
      for i,img in enumerate(imgset):
        tt.running('loading', "...on image {0}/{1}".format(i,len(imgset)))

        xml_filename = os.path.join(self.config.pascal_dir,'Annotations',img+'.xml')
        self.images.append(
          Image.load_from_pascal_xml_filename(self.classes,xml_filename,images_dir))

      print("...saving to cache file")
      data = {'classes': self.classes, 'images': self.images}
      filename = self.config.get_cached_dataset_filename(name)
      with open(filename, 'w') as f: cPickle.dump(data, f)
      
      print("...done in %.2f s\n"%tt.qtoc())
    
    return self.after_load()

  ###
  # Pos/Neg Windows
  # TODO: this stuff is out of date
  ###
  def get_pos_windows(self, cls=None, window_params=None, min_overlap=0.7):
    """
    Return array of all ground truth windows for the class, plus windows 
    that can be generated with window_params that overlap with it by more
    than min_overlap.
    * If cls not given, return positive windows for all classes.
    * If window_params not given, use default for the class.
    * Adjust min_overlap to fetch fewer windows.
    """
    sw = SlidingWindows(self, self)
    if not window_params:
      window_params = sw.get_default_window_params(cls)
    overlapping_windows = []
    image_inds = self.get_pos_samples_for_class(cls)
    times = []
    window_nums = []
    for i in image_inds:
      image = self.images[i]
      gts = image.get_ground_truth(cls)
      if gts.arr.shape[0]>0:
        overlap_wins = gts.arr[:,:4]
        overlap_wins = np.hstack((overlap_wins, np.tile(i, (overlap_wins.shape[0],1))))
        overlapping_windows.append(overlap_wins.astype(int))
        windows,time_elapsed = image.get_windows(window_params,with_time=True)
        window_nums.append(windows.shape[0])
        times.append(time_elapsed)
        for gt in gts.arr:
          overlaps = BoundingBox.get_overlap(windows[:,:4],gt[:4])
          overlap_wins = windows[overlaps>=min_overlap,:]
          overlap_wins = np.hstack((overlap_wins, np.tile(i, (overlap_wins.shape[0],1))))
          overlapping_windows.append(overlap_wins.astype(int))
          windows = windows[overlaps<min_overlap,:]
    overlapping_windows = np.concatenate(overlapping_windows,0)
    print("Windows generated per image: %d +/- %.3f, in %.3f +/- %.3f sec"%(
          np.mean(window_nums),np.std(window_nums),
          np.mean(times),np.std(times)))
    return overlapping_windows

  def get_neg_windows(self, num, cls=None, window_params=None, max_overlap=0,
      max_num_images=250):
    """
    Return array of num windows that can be generated with window_params
    that do not overlap with ground truth by more than max_overlap.
    * If cls is not given, returns ground truth for all classes.
    * If max_num_images is given, samples from at most that many images.
    """
    sw = SlidingWindows(self, self)
    if not window_params:
      window_params = sw.get_default_window_params(cls)
    all_windows = []
    image_inds = self.get_pos_samples_for_class(cls)

    max_num = len(image_inds)
    inds = image_inds
    if max_num_images:
      inds = skutil.random_subset(image_inds, max_num_images)
    num_per_image = round(1.*num / max_num)
    for ind in inds:
      image = self.images[ind]
      windows = image.get_windows(window_params)
      gts = image.get_ground_truth(cls)
      for gt in gts.arr:
        overlaps = BoundingBox.get_overlap(windows[:,:4],gt[:4])
        windows = windows[overlaps <= max_overlap,:]
      if windows.shape[0] == 0:
        continue
      ind_to_take = skutil.random_subset_up_to_N(windows.shape[0], num_per_image)
      all_windows.append(np.hstack(
        (windows[ind_to_take,:],np.tile(ind, (ind_to_take.shape[0],1)))))
    all_windows = np.concatenate(all_windows,0)
    return all_windows[:num,:]

  ###
  # Ground truth
  ###
  def get_cls_counts(self, with_diff=True, with_trun=True):
    """
    Return (N,K) array of class presence counts, where
    - n corresponds to index into self.images,
    - k corresponds to index into self.classes.
    """
    assert(len(self.images)>0)
    kwargs = {'with_diff':with_diff, 'with_trun':with_trun}
    return skutil.collect(self.images, Image.get_cls_counts, kwargs)

  def get_cls_ground_truth(self,with_diff=True,with_trun=True):
    """
    Return Table of classification (0/1) ground truth.
    """
    assert(len(self.images)>0)
    arr = self.get_cls_counts(with_diff,with_trun)>0
    return Table(arr,self.classes)

  def get_det_gt(self, with_diff=True, with_trun=True):
    """
    Return Table of detection ground truth.
    Cache the results for the given parameter settings.
    """
    assert(len(self.images)>0)
    name = '%s%s'%(with_diff,with_trun)
    if name not in self.cached_det_ground_truth:
      kwargs = {'with_diff':with_diff, 'with_trun':with_trun}
      table = skutil.collect_with_index(
        self.images, Image.get_det_gt, kwargs, 'img_ind')
      self.cached_det_ground_truth[name] = table
    return self.cached_det_ground_truth[name]

  def get_det_gt_for_class(self, class_name, with_diff=True, with_trun=True):
    """
    Return Table of detection ground truth, filtered for the given class.
    """
    assert(len(self.images)>0)
    gt = self.get_det_gt(with_diff,with_trun)
    return gt.filter_on_column('cls_ind', self.classes.index(class_name))

  ###
  # Statistics
  ###
  def plot_distribution(self):
    """
    Plot histogram of # classes in an image.

    Args:
      None

    Returns:
      - fig (plt.Figure): 

    Raises:
      None
    """
    table = self.get_cls_ground_truth(with_diff=False,with_trun=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(1,max(table.sum(1))+2)
    ax.hist(table.sum(1),bins,align='left',normed=True)
    ax.set_xticks(bins)
    ax.set_xlabel('Number of classes present in the image')
    ax.grid(False)

    dirname = self.config.get_dataset_stats_dir(self)
    filename = os.path.join(dirname,'num_classes.png')
    fig.savefig(filename)
    return fig

  def plot_coocurrence(self, cmap=plt.cm.Reds, color_anchor=[0,1],
    x_tick_rot=90, size=None, title=None, plot_vals=True,
    second_order=False):
    """
    Plot a heat map of conditional occurence, where cell (i,j) means
    P(C_j|C_i). The last column in the K x (K+2) heat map corresponds
    to the prior P(C_i).

    If second_order, plots (K choose 2) x (K+2) heat map corresponding
    to P(C_i|C_j,C_k): second-order correlations.

    Return the figure.
    """
    table = self.get_cls_ground_truth(with_diff=False,with_trun=True)

    # This takes care of most of the difference between normal and second_order
    # In the former case, a "combination" is just one class to condition on.
    combinations = combination_strs = table.cols
    if second_order: 
      combinations = [x for x in itertools.combinations(table.cols,2)]
      combination_strs = ['%s, %s'%(x[0],x[1]) for x in combinations]

    total = table.shape[0]
    N = len(table.cols)
    K = len(combinations)
    data = np.zeros((K,N+2)) # extra columns are for P("nothing"|C) and P(C)
    prior = np.zeros(K)
    for i,combination in enumerate(combinations):
      if second_order:
        cls1 = combination[0]
        cls2 = combination[1]
        conditioned = table.filter_on_column(cls1).filter_on_column(cls2)
      else:
        conditioned = table.filter_on_column(combination)

      # count all the classes
      data[i,:-2] = conditioned.sum()

      # count the number of times that cls was the only one present to get
      # P("nothing"|C)
      if second_order:
        data[i,-2] = ((conditioned.sum(1)-2)==0).sum()
      else:
        data[i,-2] = ((conditioned.sum(1)-1)==0).sum()

      # normalize
      max_val = np.max(data[i,:])
      data[i,:] /= max_val
      data[i,:][data[i,:]==1]=np.nan

      # use the max count to compute the prior
      data[i,-1] = max_val / total

      m = Table(data,table.cols+['nothing','prior'],index=combination_strs)

    # If second_order, sort by prior and remove rows with 0 prior
    if second_order:
      m = m.filter_on_column('prior',0.001,operator.gt).\
            sort_by_column('prior',descending=True)
      # TODO: just take the top K actually, for a side-by-side figure
      m.arr = m.arr[:len(self.classes),:]

    if size:
      fig = plt.figure(figsize=size)
    else:
      w=max(12,m.shape[1])
      h=max(12,m.shape[0])
      fig = plt.figure(figsize=(w,h))
    ax_im = fig.add_subplot(111)

    # make axes for colorbar
    divider = make_axes_locatable(ax_im)
    ax_cb = divider.new_vertical(size="5%", pad=0.1, pack_start=True)
    fig.add_axes(ax_cb)

    #The call to imshow produces the matrix plot:
    im = ax_im.imshow(m.arr, origin='upper', interpolation='nearest',
            vmin=color_anchor[0], vmax=color_anchor[1], cmap=cmap)

    #Formatting:
    ax = ax_im
    ax.set_xticks(np.arange(m.shape[1]))
    ax.set_xticklabels(m.cols)
    for tick in ax.xaxis.iter_ticks():
      tick[0].label2On = True
      tick[0].label1On = False
      tick[0].label2.set_rotation(x_tick_rot)
      tick[0].label2.set_fontsize('x-large')

    ax.set_yticks(np.arange(m.shape[0]))
    ax.set_yticklabels(m.index,size='x-large')

    ax.yaxis.set_minor_locator(
      matplotlib.ticker.FixedLocator(np.arange(-.5,m.shape[0]+0.5)))
    ax.xaxis.set_minor_locator(
      matplotlib.ticker.FixedLocator(np.arange(-.5,m.shape[1]-0.5)))
    ax.grid(False,which='major')
    ax.grid(True,which='minor',ls='-',lw=7,c='w')

    # Make the major and minor tick marks invisible
    for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        line.set_markeredgewidth(0)
    for line in ax.xaxis.get_minorticklines() + ax.yaxis.get_minorticklines():
        line.set_markeredgewidth(0)

    # Limit the area of the plot
    ax.set_ybound([-0.5, m.shape[0] - 0.5])
    ax.set_xbound([-0.5, m.shape[1] - 0.5])

    #The following produces the colorbar and sets the ticks
    #Set the ticks - if 0 is in the interval of values, set that, as well
    #as the maximal and minimal values:
    #Extract the minimum and maximum values for scaling
    max_val = np.nanmax(m.arr)
    min_val = np.nanmin(m.arr)
    if min_val < 0:
      ticks = [color_anchor[0], min_val, 0, max_val, color_anchor[1]]
    #Otherwise - only set the maximal value:
    else:
      ticks = [color_anchor[0], max_val, color_anchor[1]]

    # Plot line separating 'nothing' and 'prior' from rest of plot
    l = ax.add_line(mpl.lines.Line2D(
      [m.shape[1]-2.5,m.shape[1]-2.5],[-.5,m.shape[0]-0.5],
      ls='--',c='gray',lw=2))
    l.set_zorder(3)

    # Display the actual values in the cells
    if plot_vals:
      for i in xrange(0, m.shape[0]):
        for j in xrange(0,m.shape[1]):
          val = m.arr[i,j]
          if np.isnan(val):
            continue
          if val > 0.5:
            ax.text(j-0.2,i+0.1,'%.2f'%val,color='w')
          else:
            ax.text(j-0.2,i+0.1,'%.2f'%val,color='k')

    # Hide the black frame around the plot
    # Doing ax.set_frame_on(False) results in weird thin lines
    # from imshow() at the edges. Instead, we set the frame to white.
    for spine in ax.spines.values():
      spine.set_edgecolor('w')

    # Set title
    if title is not None:
      ax.set_title(title)

    # Plot the colorbar and remove its frame as well.
    cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal',
            cmap=cmap, ticks=ticks, format='%.2f')
    cb.ax.artists.remove(cb.outline)

    # Save figure
    dirname = self.config.get_dataset_stats_dir(self)
    suffix = '_second_order' if second_order else ''
    filename = os.path.join(dirname,'cooccur%s.png'%suffix)
    fig.savefig(filename)

    return fig

  ###
  # K-Folds
  ###
  def create_folds(self, numfolds):
    """
    Split the images of dataset in numfolds folds.
    Dataset has an inner state about current fold (This is like an implicit 
    generator).
    """
    self.folds = [fold for fold in KFold(len(self.images), numfolds)]
    self.current_fold = 0
    
  def next_folds(self):
    # TODO: check that correct
    if self.current_fold < len(self.folds):
      fold = self.folds[self.current_fold]
      self.current_fold += 1
      self.train, self.val = fold
      if type(self.train[0]) == type(np.array([True])[0]):
        self.train = np.where(self.train)[0]
        self.val = np.where(self.val)[0]
      return fold
    else:
      self.current_fold = 0
      return 0
  
  def get_fold_by_index(self, ind):
    """
    Random access to folds
    """
    if ind >= len(self.folds):
      raise RuntimeError('Try to access non-existing fold')
    else:
      return self.folds[ind]
  
  def get_pos_samples_for_fold_class(self, cls, with_diff=False,
      with_trun=True):
    # TODO: reimplement
    if not hasattr(self, 'train'):
      return self.get_pos_samples_for_class(cls, with_diff, with_trun)
    all_pos = self.get_pos_samples_for_class(cls, with_diff, with_trun)
    return np.intersect1d(all_pos, self.train)
  
  def get_neg_samples_for_fold_class(self, cls, num_samples, with_diff=False,
      with_trun=True):
    if num_samples==0:
      return np.array([])
    if not hasattr(self, 'train'):
      return self.get_neg_samples_for_class(cls, num_samples, with_diff, with_trun)
    all_neg = self.get_neg_samples_for_class(cls, None, with_diff, with_trun)
    intersect = np.intersect1d(all_neg, self.train)
    if intersect.size == 0:
      return np.array([])
    return np.array(skutil.random_subset(intersect, num_samples))

  def get_pos_samples_for_class(self, cls, with_diff=False,
      with_trun=True):
    """
    Return array of indices of self.images that contain at least one object of
    this class.
    """
    # TODO: this can be much faster! dont use det_gt
    cls_gt = self.get_det_gt_for_class(cls,with_diff,with_trun)
    img_indices = cls_gt.subset_arr('img_ind')
    return np.sort(np.unique(img_indices)).astype(int)

  def get_neg_samples_for_class(self, cls, number=None,
      with_diff=False, with_trun=True):
    """
    Return array of indices of self.images that contain no objects of this class.
    """
    if number == 0:
      return np.array([])
    pos_indices = self.get_pos_samples_for_class(cls,with_diff,with_trun)
    neg_indices = np.setdiff1d(np.arange(len(self.images)),pos_indices,assume_unique=True)
    return skutil.random_subset(neg_indices, number)

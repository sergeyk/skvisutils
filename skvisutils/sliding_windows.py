import time
import numpy as np

from skvisutils import BoundingBox

class WindowParams:
  """
  Encapsulate parameters that are needed to generate windows.
  NOTE: be careful to only set these to lists and not ndarrays!
  """

  def __init__(self,
      min_width=10,
      stride=8,
      scales=[1., 0.70710678, 0.5, 0.35355339, 0.25],
      aspect_ratios=[1/2., 2/3., 1., 3/2., 2.]):
    self.min_width = int(min_width)
    self.stride = int(stride)
    self.scales = [x for x in scales]
    self.aspect_ratios = [x for x in aspect_ratios]

  def __repr__(self):
    return  "min_width: %s, stride: %s\n"%(self.min_width,self.stride) + \
            "scales: %s\naspect_ratios: %s"%(self.scales,self.aspect_ratios)

  @classmethod
  def load_from_json(cls,filename):
    """Return WindowParams object loaded from JSON file."""
    wp = WindowParams()
    with open(filename) as f:
      wp.__dict__ = json.load(f)
    return wp

  def save_to_json(self,filename):
    """Serializes self as a JSON file."""
    with open(filename,'w') as f:
      json.dump(self.__dict__,f)

class SlidingWindows:
  """
  Encapsulate generating window proposals for a 'sliding' window classifier.
  Analyze statistics of ground truth windows to find the best parameters for
  generating windows.
  """

  MIN_WIDTH=10 #px

  def __init__(self, dataset, train_dataset):
    """
    Initialize with dataset whose statistics we will aggregate.
    Statistics are not initially computed; they are computed and cached on the
    first call to get_stats().
    """
    self.dataset = dataset
    self.train_dataset = train_dataset
    self.train_name = self.train_dataset.get_name()
    self.stats = None
    self.cached_params = {}
    self.jw = None

  def get_stats(self):
    """
    Compute, cache, and return the statistics of window distributions.
    """
    if not self.stats:
      self.stats = SlidingWindows.get_dataset_window_stats(self.train_dataset)
    return self.stats

  def get_default_window_params(self,cls=None,mode='fastest_good'):
    """
    Return default setting of window params for the given cls.
    These are set from data and a grid search over parameters is done to find
    the best combination.
    If cls is None, returns window settings suitable for all classes.
    """
    if not cls:
      cls = 'all'
    key = '%s_%s_%s'%(0.5,mode,cls)
    if key not in self.cached_params:
      filename = config.get_window_params_json(self.train_name)%key
      window_params = WindowParams.load_from_json(filename)
      self.cached_params[key] = window_params
    return self.cached_params[key]

  @classmethod
  def get_recall_vs_num_auc(cls,table):
    """
    Return tuple of 
      - the fraction of total possible logarithmic area captured under the curve
        The rationale is that the number becomes less meaningful as the total
        area dwarfs the variation.
      - the recall at 1K windows
    """
    num_windows = table.subset_arr('num_windows')
    recall = table.subset_arr('recall')
    auc = np.sum(recall)/recall.size
    max_ind = np.flatnonzero(num_windows<=1000)[-1]
    max_rec = recall[max_ind]
    return (auc,max_rec)

  @classmethod
  def plot_recall_vs_num(cls,window_tables,filename,force=True):
    """
    Take list of Tables containing [num_windows,recall] data.
    These Tables are generated by SlidingWindows.evaluate_recall.
    Legend entries are composed from .name entries in the Tables.
    Save plot to given filename.
    Does not return anything.
    """
    if os.path.exists(filename) and not force:
      return
    colors = ['orange','black','#4084ff']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_num_windows = -1
    for i,table in enumerate(window_tables): 
      num_windows = table.subset_arr('num_windows')
      if num_windows[-1]>max_num_windows:
        max_num_windows = num_windows[-1]
      recall = table.subset_arr('recall')
      # compute our metric: the fraction of total possible area captured under
      # the curve
      auc,max_rec = SlidingWindows.get_recall_vs_num_auc(table)
      label = "%s: auc=%.2f, rec=%.2f"%(table.name,auc,max_rec)
      ax.plot(num_windows,recall,'-s',
          color=colors[i],linewidth=2,label=label)
    plt.legend()
    plt.xscale('symlog')
    plt.xlim(0,max_num_windows)
    plt.legend(loc='lower right')
    plt.xlabel('Number of windows',size=15)
    plt.ylabel('Recall',size=15)
    ax.xaxis.grid(True)
    plt.yticks(np.arange(0,1.1,0.1))
    ax.yaxis.grid(True,which='both')
    fig.savefig(filename)

  @classmethod
  def get_windows(clas,image,cls=None,window_params=None,with_time=False):
    """
    Return all windows that can be generated with window_params.
    If with_time=True, return tuple of (windows, time_elapsed).
    """
    assert(cls or window_params)
    if not window_params:
      window_params = self.get_default_window_params(cls)
    t = time.time()
    stride = window_params.stride
    min_width = window_params.min_width
    actual_xs = []
    actual_ys = []
    actual_ws = []
    actual_hs = []
    num_windows = 0
    # we want to be able to capture objects that extend past the image
    # we always iterate over locations in native space, and convert to
    # actual image space when we record the window
    w_pad = int(1.*min_width/2)
    x_min = -w_pad
    for scale in window_params.scales:
      x_max = int(image.width*scale)-w_pad
      if w_pad > 0:
        x_max += stride
      actual_w = int(min_width/scale) + 1
      for ratio in window_params.aspect_ratios:
        h_pad = int(1.*min_width*ratio/2)
        y_min = -h_pad
        y_max = int(image.height*scale)-h_pad
        if h_pad > 0:
          y_max += stride
        actual_h = int(min_width/scale * ratio) + 1
        for y in range(y_min,y_max,stride):
          for x in range(x_min,x_max,stride):
            actual_ws.append(actual_w)
            actual_hs.append(actual_h)
            actual_xs.append(int(x/scale))
            actual_ys.append(int(y/scale))
    windows = np.array([actual_xs,actual_ys,actual_ws,actual_hs]).T
    windows = BoundingBox.clipboxes_arr(windows,(0,0,image.width,image.height))
    if with_time:
      time_elapsed = time.time()-t
      return (windows,time_elapsed)
    else:
      return windows
  
  def grid_search_over_metaparams(self):
    """
    Evaluates different metaparams for the get_windows_new method with the
    metric of AUC under the recall vs. # windows curve.
    """
    # approximately strides of 4, 6, and 8 px
    samples_per_500px_vals = [62, 83, 125]
    num_scales_vals = [9,12,15]
    num_ratios_vals = [6,9,12]
    mode_vals = ['linear','importance']
    priority_vals = [0]
    classes = self.dataset.classes+['all']

    dirname = config.get_sliding_windows_metaparams_dir(self.train_name)
    table_filename = os.path.join(dirname,'table.csv')
    if os.path.exists(table_filename):
      table = Table.load_from_csv(table_filename)
    else:
      grid_vals = [x for x in itertools.product(
        samples_per_500px_vals,num_scales_vals,num_ratios_vals,mode_vals,priority_vals,classes)]
      num_combinations = len(grid_vals)
      print("Running over %d combinations of class and parameters."%num_combinations)

      cols = ['cls_ind','samples_per_500px','num_scales','num_ratios','mode_ind','priority','complexity','auc','max_recall']
      grid = np.zeros((num_combinations,len(cols)))
      for i in range(comm_rank, num_combinations, comm_size):
        grid_val = grid_vals[i]
        print(grid_val)
        samples,num_scales,num_ratios,mode,priority,cls = grid_val
        cls_ind = self.dataset.get_ind(cls)

        metaparams = {
          'samples_per_500px': samples,
          'num_scales': num_scales,
          'num_ratios': num_ratios,
          'mode': mode,
          'priority': priority}
        mode_ind = mode_vals.index(metaparams['mode'])

        filename = '%s_%d_%d_%d_%s_%d'%(
            cls,
            metaparams['samples_per_500px'],
            metaparams['num_scales'],
            metaparams['num_ratios'],
            metaparams['mode'],
            metaparams['priority'])
        cls_dirname = os.path.join(dirname,cls)
        ut.makedirs(cls_dirname)
        filename = os.path.join(cls_dirname,filename)

        tables = self.evaluate_recall(cls,filename,metaparams,'sw',plot=False,force=False)

        # compute the final metrics 
        auc,max_rec = SlidingWindows.get_recall_vs_num_auc(tables[1]) # the ov=0.5 table
        complexity = samples*num_scales*num_ratios
        grid[i,:] = np.array([cls_ind, samples, num_scales, num_ratios, mode_ind, priority, complexity, auc, max_rec])
      # Reduce the MPI jobs
      if comm_rank == 0:
        grid_all = np.zeros((num_combinations,len(cols)))
      else:
        grid_all = None
      safebarrier(comm)
      comm.Reduce(grid,grid_all)
      table = Table(grid_all,cols)
      table.save_csv(table_filename)

    # print the winning parameters in the table
    for cls in self.dataset.classes+['all']:
      st = table.filter_on_column('cls_ind',self.dataset.get_ind(cls))
      aucs = st.subset_arr('auc')
      max_recalls = st.subset_arr('max_recall')
      best_auc_ind = aucs.argmax()
      best_max_recall_ind = max_recalls.argmax()
      print("%s: best AUC is %.3f with metaparams (%d, %d, %d, %s, %d)"%(
        cls, aucs[best_auc_ind],
        st.arr[best_auc_ind,1], st.arr[best_auc_ind,2], st.arr[best_auc_ind,3],
        mode_vals[int(st.arr[best_auc_ind,4])], st.arr[best_auc_ind,5]))
      print("%s: best max recall is %.3f with metaparams (%d, %d, %d, %s, %d)"%(
        cls, max_recalls[best_max_recall_ind],
        st.arr[best_max_recall_ind,1], st.arr[best_max_recall_ind,2],
        st.arr[best_max_recall_ind,3],
        mode_vals[int(st.arr[best_max_recall_ind,4])], st.arr[best_max_recall_ind,5]))

      complexities = st.subset_arr('complexity')
      complexities /= complexities.max()
      d_max_recalls = max_recalls/complexities
      d_aucs = aucs/complexities
      best_auc_ind = d_aucs.argmax()
      best_max_recall_ind = d_max_recalls.argmax()
      print("%s: best AUC/complexity is %.3f with metaparams (%d, %d, %d, %s, %d)"%(
        cls, aucs[best_auc_ind],
        st.arr[best_auc_ind,1], st.arr[best_auc_ind,2], st.arr[best_auc_ind,3],
        mode_vals[int(st.arr[best_auc_ind,4])], st.arr[best_auc_ind,5]))
      print("%s: best max recall/complexity is %.3f with metaparams (%d, %d, %d, %s, %d)"%(
        cls, max_recalls[best_max_recall_ind],
        st.arr[best_max_recall_ind,1], st.arr[best_max_recall_ind,2],
        st.arr[best_max_recall_ind,3],
        mode_vals[int(st.arr[best_max_recall_ind,4])], st.arr[best_max_recall_ind,5]))
    return table

  def evaluate_recall(self,cls,filename,metaparams,mode,plot=False,force=False):
    """Plot recall vs. num_windows."""
    t = time.time()
    if os.path.exists(filename+'.npy') and not force:
      tables = np.load(filename+'.npy')
    else:
      if mode=='jw':
        #window_intervals = [0, 3, 10, 35, 100, 300, 1000, 2000, 4000, 8000, 16000]
        window_intervals = logspace(0,4,9).astype(int)
        # array([    1,     3,    10,    31,   100,   316,  1000,  3162, 10000])
      elif mode=='sw':
        #window_intervals = [0, 3, 10, 35, 100, 300, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]
        window_intervals = logspace(0,6,13).astype(int)
        # array([      1,       3,      10,      31,     100,     316,    1000,
        #  3162,   10000,   31622,  100000,  316227, 1000000])
      else:
        raise RuntimeError("Impossible mode")
      min_overlaps=[0.25,0.5,0.75]
      recalls = self.get_recalls(cls,metaparams,mode,window_intervals,min_overlaps)

      # Save recalls to Table and call plotting method
      cols = ['num_windows','recall']
      tables = []
      for min_overlap in min_overlaps:
        arr = np.array((np.array(window_intervals),
          recalls[:,min_overlaps.index(min_overlap)])).T
        label = '%s, %s, ov=%.2f'%(cls,mode,min_overlap)
        tables.append(Table(arr,cols,label))
        np.save(filename,tables)
    if plot:
      try:
        SlidingWindows.plot_recall_vs_num(tables,filename,force=force)
      except (KeyboardInterrupt, SystemExit):
        raise
      except Exception, e:
        print(e)
        print("Probably can't plot because we're using mpirun")
    print("Evaluating recall took %.3f s"%(time.time()-t))
    return tables

  def get_recalls(self,cls,metaparams,mode,window_intervals,min_overlaps):
    """
    Return nparray of num_intervals x num_overlaps, with each entry specifying
    the recall for that combination of window_interval and min_overlap.
    window_intervals must begin with 0.
    mode must be in ['sw','jw']
    """
    assert(window_intervals[0] == 0)
    num_overlaps = len(min_overlaps)
    num_intervals = len(window_intervals)
    times = [0]
    window_nums = [0]
    image_inds = self.dataset.get_pos_samples_for_class(cls)
    num_images = len(image_inds)
    # we are building up a num_images x num_intervals+1 x num_overlaps array
    array = np.zeros((num_images,num_intervals+1,num_overlaps))
    for i in range(num_images):
      ind = image_inds[i]
      image = self.dataset.images[ind]
      # the first interval is 0, so there aren't any window proposals 
      array[i,0,:] = 0
      gts = image.get_ground_truth(cls)
      num_gt = gts.shape[0]
      # the last row of the matrix is the number of ground truth
      array[i,num_intervals,:] = num_gt
      # now get the windows and append the statistics information
      #windows,time_elapsed = window_generator.get_windows(image,cls,with_time=True)

      if mode=='sw': 
        windows,time_elapsed = self.get_windows_new(image,cls,metaparams,with_time=True,at_most=max(window_intervals))
      elif mode=='jw':
        windows,time_elapsed = self.jw.get_windows(image,cls,K=10000)
      else:
        raise RuntimeError('impossible mode')

      # shuffle the windows if we want to take them in random order
      if mode=='sw' and not metaparams['priority']:
        rand_ind = np.random.permutation(windows.shape[0])
        windows = windows[rand_ind,:]

      window_nums.append(windows.shape[0])
      times.append(time_elapsed)
      # go through each interval and count how many ground truth are matched
      for j in range(1,len(window_intervals)):
        max_ind = window_intervals[j]
        # if we are going to ask for more windows that are available,
        # the recall is going to be the same as before, so just add that
        if max_ind>windows.shape[0]:
          array[i,j,:] = array[i,j-1,:]
          continue
        # otherwise, count the number of ground truths that are overlapped
        # NOTE: a single window can overlap multiple ground truth in this
        # scheme
        for gt in gts.arr:
          overlaps = BoundingBox.get_overlap(windows[:max_ind,:4],gt[:4])
          for k,min_overlap in enumerate(min_overlaps):
            if np.any(overlaps>=min_overlap):
              array[i,j,k] += 1
    print("Windows generated per image: %d +/- %.3f, in %.3f +/- %.3f sec"%(
      np.mean(window_nums),np.std(window_nums),
      np.mean(times),np.std(times)))
    # reduce to num_intervals+1 x num_overlaps
    sum_array = np.sum(array,axis=0)
    # reduce to num_intervals x num_overlaps
    recalls = sum_array[:-1,:]/sum_array[-1,:]
    return recalls

  def get_window_params_for_cls(self, cls,
      stride=6, num_ar_points=10, num_scale_points=12, mode='importance'):
    """
    Return WindowParams object with the default min_width, given stride, and
    the sampling points given the two parameters of num_X_points.
    The sampling points are determined from data:
      - if mode=='importance', samples according to object likelihood
      - if mode=='linear', samples linearly from data min to max.
    """
    stats = self.get_stats()
    aspect_ratios = stats['%s_aspect_ratios'%cls]
    scales = stats['%s_scales'%cls]
    wp = WindowParams()
    wp.min_width = self.MIN_WIDTH
    wp.stride = stride
    if mode=='importance':
      # NOTE: Answers are slightly different depending on whether we're
      # sampling on a log scale.
      wp.aspect_ratios = np.exp(ut.importance_sample(np.log(aspect_ratios),num_ar_points)).tolist()
      wp.scales = ut.sample_uniformly(scales,num_scale_points).tolist()
    elif mode=='linear':
      wp.aspect_ratios = np.linspace(np.min(aspect_ratios),np.max(aspect_ratios)).tolist()
      wp.scales = np.linspace(np.min(scales),np.max(scales)).tolist()
    else:
      raise Exception("unknown mode")
    return wp

  ###############
  # Statistics of ground truth
  ###############
  @classmethod
  def get_dataset_window_stats(cls,dataset,plot=False,force=False):
    """
    Return the statistics of ground truth window parameters: x,y,scale,ratio.
    If plot=True, write plots out to files.
    If force=False, do not overwrite files that exist.
    """
    print("SlidingWindows: getting stats of the %s dataset"%dataset.get_name())
    t = time.time()
    results_file = config.get_window_stats_results(dataset.get_name())
    if os.path.exists(results_file) and not (force or plot):
      with open(results_file) as f:
        results = cPickle.load(f)
    else:
      results = {}
      for cls in dataset.classes + ['all']:
        cls_gt_table = dataset.get_ground_truth_for_class(cls,with_diff=False)
        bboxes = cls_gt_table.subset_arr(['x','y','w','h'])
        if bboxes.shape[0]<1:
          continue

        img_inds = cls_gt_table.subset_arr('img_ind').astype(int)
        images = [dataset.images[i] for i in img_inds]
        image_widths = [img.width for img in images]
        image_heights = [img.height for img in images]

        # scale = width/min_width
        scale =  1.*bboxes[:,2] / SlidingWindows.MIN_WIDTH
        scale_expanded = SlidingWindows.expand_dist(scale,bounds=(1,50)) 
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'scale',cls)
          SlidingWindows.plot_dist(scale,filename,scale_expanded,
              xlim=[0,55], xlabel='scale',title=cls,force=force)

        # x_scaled = x/(img_width/scale) = x*scale/img_width
        x_scaled = 1.*bboxes[:,0] * scale / image_widths
        x_scaled_expanded = SlidingWindows.expand_dist(x_scaled)
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'x_scaled',cls)
          SlidingWindows.plot_dist(x_scaled,filename,x_scaled_expanded,
              xlim=[-2,15], xlabel='x as a fraction of the scaled width',title=cls,force=force)

        # x_frac = x/img_width
        x_frac = 1.*bboxes[:,0] / image_widths
        x_frac_expanded = SlidingWindows.expand_dist(x_frac,bounds=(0,1))
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'x_frac',cls)
          SlidingWindows.plot_dist(x_frac,filename,x_frac_expanded,
              xlim=[-0.2,1.2],xlabel='x as a fraction of the width',title=cls,force=force)

        # y_scaled = y*y_scale/img_height = y*(height/min_width)/img_height
        y_scaled = 1.*bboxes[:,1]*bboxes[:,3]/SlidingWindows.MIN_WIDTH/image_heights
        y_scaled_expanded = SlidingWindows.expand_dist(y_scaled)
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'y_scaled',cls)
          SlidingWindows.plot_dist(y_scaled,filename,y_scaled_expanded,
              xlim=[-2,15],xlabel='y as a fraction of the scaled height',title=cls,force=force)

        # y_frac = y / img_height
        y_frac = 1.*bboxes[:,1]/image_heights
        y_frac_expanded = SlidingWindows.expand_dist(y_frac,bounds=(0,1))
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'y_frac',cls)
          SlidingWindows.plot_dist(y_frac,filename,y_frac_expanded,
              xlim=[-0.2,1.2],xlabel='y as a fraction of the frac height',title=cls,force=force)

        # ratio = height/width
        log_ratio = np.log(1.*bboxes[:,3]/bboxes[:,2])
        log_ratio_expanded = SlidingWindows.expand_dist(log_ratio)
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'log_ratio',cls)
          SlidingWindows.plot_dist(log_ratio,filename,log_ratio_expanded,
              xlim=[-2.5,2.5],xlabel='log of aspect ratio',title=cls,force=force)
        
        # plot x_scaled vs y_scaled:
        dists = np.array((x_scaled,y_scaled))
        expanded_dists = np.array((x_scaled_expanded,y_scaled_expanded))
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'x_scaled_vs_y_scaled',cls)
          SlidingWindows.plot_two_dists(dists,filename,expanded_dists,
              xlim=[-2,15],ylim=[-2,15],
              xlabel='x as a fraction of the scaled width',
              ylabel='y as a fraction of the scaled height',
              title=cls,force=force)

        # plot x_frac vs y_frac:
        dists = np.array((x_frac,y_frac))
        expanded_dists = np.array((x_frac_expanded,y_frac_expanded))
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'x_frac_vs_y_frac',cls)
          SlidingWindows.plot_two_dists(dists,filename,expanded_dists,
              xlim=[-0.2,1.2],ylim=[-0.2,1.2],
              xlabel='x as a fraction of the width',
              ylabel='y as a fraction of the height',
              title=cls,force=force)

        # plot x_scaled vs scale:
        dists = np.array((x_frac,scale))
        expanded_dists = np.array((x_frac_expanded,scale_expanded))
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'x_frac_vs_scale',cls)
          SlidingWindows.plot_two_dists(dists,filename,expanded_dists,
              xlim=[-0.2,1.2],ylim=[0,55],
              xlabel='x as a fraction of the width',
              ylabel='scale',
              title=cls,force=force)

        # plot x_scaled vs scale:
        dists = np.array((x_scaled,scale))
        expanded_dists = np.array((x_scaled_expanded,scale_expanded))
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'x_scaled_vs_scale',cls)
          SlidingWindows.plot_two_dists(dists,filename,expanded_dists,
              xlim=[-2,15],ylim=[0,55],
              xlabel='x as a fraction of the scaled width',
              ylabel='scale',
              title=cls,force=force)

        # plot log_ratio vs scale:
        dists = np.array((log_ratio,scale))
        expanded_dists = np.array((log_ratio_expanded,scale_expanded))
        if plot:
          filename = config.get_window_stats_plot(dataset.get_name(),'log_ratio_vs_scale',cls)
          SlidingWindows.plot_two_dists(dists,filename,expanded_dists,
              xlim=[-2.5,2.5],ylim=[0,55],
              xlabel='log of aspect ratio',
              ylabel='scale',
              title=cls,force=force)
        
        # Assemble univariate KDEs--downsampled for large-data classes
        num_points = scale.size
        max_points = 500
        rand_ind = ut.random_subset_up_to_N(num_points, max_points)
        kde = st.gaussian_kde(log_ratio_expanded[rand_ind])
        results['%s_%s_kde'%(cls,'log_ratio')] = kde
        kde = st.gaussian_kde(y_frac_expanded[rand_ind])
        results['%s_%s_kde'%(cls,'y_frac')] = kde
        kde = st.gaussian_kde(x_frac_expanded[rand_ind])
        results['%s_%s_kde'%(cls,'x_frac')] = kde
        kde = st.gaussian_kde(scale[rand_ind])
        results['%s_%s_kde'%(cls,'scale')] = kde
            
        # Assemble multivariate kde
        # TODO: do I need to whiten this data or something?
        dist = np.array([x_frac_expanded,y_frac_expanded,scale_expanded,log_ratio_expanded])
        kde = st.gaussian_kde(dist)
        results['%s_kde'%cls] = kde

      # Save the results info
      with open(results_file,'w') as f:
        cPickle.dump(results,f)

    print("SlidingWindows: took %.3f sec"%(time.time()-t))
    return results

  @classmethod
  def expand_dist(cls,dist,bounds=None,margin=0.15):
    """
    Expand the given distribution by margin*range of the data, by appending two
    extra datapoints at the end.
    If bounds are given, does not go beyond them--but still appends the extra
    points.
    """
    # TODO: can improve by adding not just 2 but some number proportional to
    # the number of points already in vector.
    expanded_dist = np.zeros(len(dist)+2)
    expanded_dist[:len(dist)] = dist
    rang = np.max(dist)-np.min(dist)
    expanded_dist[-2] = np.min(dist)-margin*rang
    expanded_dist[-1] = np.max(dist)+margin*rang
    if bounds:
      expanded_dist[-2] = max(expanded_dist[-2],bounds[0])
      expanded_dist[-1] = min(expanded_dist[-1],bounds[1])
    return expanded_dist

  @classmethod
  def plot_two_dists(cls,dists,filename,expanded_dists,
      xlim=None,xlabel=None,ylim=None,ylabel=None,title=None,force=False):
    """
    Helper method to plot a scatterplot with an overlaid Gaussian KDE contour
    map and write it out to file.
    dists[0,:] will be plotted on the x axis and dists[1,:] on the y axis
    If expanded_*_dist is given, plot it as well and use it to calculate KDE.
    """
    if not force and os.path.exists(filename):
      return
    plt.clf()
    if expanded_dists == None:
      expanded_dists = dists
    else:
      plt.scatter(expanded_dists[0,:],expanded_dists[1,:],facecolor='orange',label='expanded')
    plt.scatter(dists[0,:],dists[1,:],facecolor='#4084ff',label='original')

    # Regular grid to evaluate kde upon
    num_points = 100
    x_flat = np.linspace(expanded_dists[0,:].min(), expanded_dists[0,:].max(), num_points)
    y_flat = np.linspace(expanded_dists[1,:].min(), expanded_dists[1,:].max(), num_points)
    x,y = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

    # construct and evaluate kde
    kde = st.gaussian_kde(expanded_dists)
    z = kde(grid_coords.T).reshape(num_points,num_points)
    plt.contour(x,y,z,color='k',linewidth=3,label='Gaussian KDE')

    if xlim:
      plt.xlim(xlim)
    if xlabel:
      plt.xlabel(xlabel,size='large')
    if ylim:
      plt.ylim(ylim)
    if ylabel:
      plt.ylabel(ylabel,size='large')
    plt.legend()
    if title:
      plt.title(title, size='large')
    plt.savefig(filename)

  @classmethod
  def plot_dist(cls,dist,filename,expanded_dist=None,num_bins=30,
      xlim=None,xlabel=None,title=None,force=False):
    """
    Helper method to plot a histogram with an overlaid Gaussian KDE curve
    and write it out to file.
    If expanded_dist is given, plot it and use it to calculate KDE.
    """
    if not force and os.path.exists(filename):
      return
    plt.clf()
    if expanded_dist == None:
      expanded_dist = dist
    else:
      plt.hist(expanded_dist,num_bins,normed=True, facecolor='orange', label='expanded')
    plt.hist(dist,num_bins,normed=True, facecolor='#4084ff', label='original')
    
    # construct and evaluate kde
    kde = st.gaussian_kde(expanded_dist.T)
    x = np.linspace(np.min(expanded_dist),np.max(expanded_dist))
    plt.plot(x, kde.evaluate(x), color='black', linewidth=3, label='Gaussian KDE')
    
    # see how limiting the number of data points to 250 affects the KDE
    num_points = expanded_dist.size
    max_points = 500
    rand_ind = np.random.permutation(num_points)[:max_points]
    kde = st.gaussian_kde(expanded_dist[rand_ind].T)
    plt.plot(x, kde.evaluate(x), color='#1548A6', linewidth=3, label='Downsampled Gaussian KDE')

    if xlim:
      plt.xlim(xlim)
    if xlabel:
      plt.xlabel(xlabel,size='large')
    plt.legend()
    if title:
      plt.title(title, size='large')
    plt.savefig(filename)

  def get_windows_new(self, image, cls, metaparams=None, with_time=False, at_most=200000, force=False):
    """
    Generate windows by using ground truth window stats and metaparams.
    metaparams must contain keys 'samples_per_500px', 'num_scales', 'num_ratios', 'mode'
    metaparams['mode'] can be 'linear' or 'importance' and refers to the method
    of sampling intervals per window parameter.
    If with_time=True, return tuple of (windows, time_elapsed).
    """
    if not metaparams:
      metaparams = {
        'samples_per_500px': 83,
        'num_scales': 12,
        'num_ratios': 6,
        'mode': 'importance',
        'priority': 0}

    t = time.time()
    x_samples = int(image.width/500. * metaparams['samples_per_500px'])
    y_samples = int(image.height/500. * metaparams['samples_per_500px'])

    # check for cached windows and return if found
    dirname = config.get_sliding_windows_cached_dir(self.train_name)
    filename = '%s_%d_%d_%s_%s_%d_%d_%d.npy'%(
        cls,
        metaparams['samples_per_500px'],
        metaparams['num_scales'],
        metaparams['num_ratios'],
        metaparams['mode'],
        metaparams['priority'],
        x_samples, y_samples)
    filename = os.path.join(dirname,filename)
    if os.path.exists(filename) and not force:
      windows = np.load(filename)
    else:
      # fine, we'll figure out the windows again
      # load the kde for x_scaled,y_scaled,scale,log_ratio
      stats = self.get_stats() 
      kde = stats['%s_kde'%cls]
      x_frac = kde.dataset[0,:]
      y_frac = kde.dataset[1,:]
      scale = kde.dataset[2,:]
      log_ratio = kde.dataset[3,:]

      # given the metaparameters, sample points to generate the complete list of
      # parameter combinations
      if metaparams['mode'] == 'linear':
        x_points = np.linspace(x_frac.min(),x_frac.max(),x_samples)
        y_points = np.linspace(y_frac.min(),y_frac.max(),y_samples)
        scale_points = np.linspace(scale.min(),scale.max(),metaparams['num_scales'])
        ratio_points = np.linspace(log_ratio.min(),log_ratio.max(),metaparams['num_ratios'])
      elif metaparams['mode'] == 'importance':
        x_points = ut.importance_sample(x_frac,x_samples,stats['%s_%s_kde'%(cls,'x_frac')])
        y_points = ut.importance_sample(y_frac,y_samples,stats['%s_%s_kde'%(cls,'y_frac')])
        scale_points = ut.importance_sample(scale,metaparams['num_scales'],stats['%s_%s_kde'%(cls,'scale')])
        ratio_points = ut.importance_sample(log_ratio,metaparams['num_ratios'],stats['%s_%s_kde'%(cls,'log_ratio')])
      else:
        raise RuntimeError("Invalid mode")

      combinations = [x for x in itertools.product(x_points,y_points,scale_points,ratio_points)]
      combinations = np.array(combinations).T
      
      # only take the top-scoring detections
      if metaparams['priority']:
        t22=time.time()
        scores = kde(combinations) # (so slow!)
        print("kde took %.3f s"%(time.time()-t22))
        sorted_inds = np.argsort(-scores)
        max_num = min(at_most,sorted_inds.size)
        combinations = combinations[:,sorted_inds[:max_num]]

      # convert to x,y,scale,ratio,w,h
      scale = combinations[2,:]
      # x = x_frac*img_width
      x = combinations[0,:]*img_width
      # ratio = exp(log_ratio)
      ratio = np.exp(combinations[3,:])
      # y = y_frac*img_height
      y = combinations[1,:]*img_height
      # w = scale*min_width
      w = scale*SlidingWindows.MIN_WIDTH
      # h = w*ratio
      h = w * ratio

      combinations[0,:] = x
      combinations[1,:] = y
      combinations[2,:] = w
      combinations[3,:] = h
      windows = combinations.T
      windows = BoundingBox.clipboxes_arr(windows,(0,0,img_width,img_height))
      np.save(filename,windows) # does not take more than 0.5 sec even for 10**6 windows

    time_elapsed = time.time()-t
    print("get_windows_new() got %d windows in %.3fs"%(windows.shape[0],time_elapsed))
    if with_time:
      return (windows,time_elapsed)
    else:
      return windows


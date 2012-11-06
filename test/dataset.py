from context import *
from skpyutils import Table, skutil

from skvisutils.dataset import Dataset

test_support_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'support'))
test_data1 = skutil.opjoin(test_support_dir,'data1.json')
test_data2 = skutil.opjoin(test_support_dir,'data2.json')

class TestDatasetJson:
  def setup(self):
    self.d = Dataset('test_data1')
    self.d.load_from_json(test_data1)
    self.classes = ["A","B","C"]

  def test_load(self):
    assert(self.d.num_images() == 4)
    assert(self.d.classes == self.classes)

  def test_get_det_gt(self):
    gt = self.d.get_det_gt(with_diff=True,with_trun=False)
    df = Table(
      np.array([[ 0.,  0.,  0.,  0.,  0.,  0, 0, 0.],
       [ 1.,  1.,  1.,  1.,  1.,  0, 0, 0.],
       [ 1.,  1.,  1.,  0.,  0.,  0, 0, 1.],
       [ 0.,  0.,  0.,  0.,  1.,  0, 0, 2.],
       [ 0.,  0.,  0.,  0.,  2.,  0, 0, 3.],
       [ 1.,  1.,  1.,  1.,  2.,  0, 0, 3.]]),
       ['x','y','w','h','cls_ind','diff','trun','img_ind'])
    print(gt)
    print(df)
    assert(gt == df)

  def test_get_cls_counts_json(self):
    arr = np.array(
      [ [ 1, 1, 0],
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 2]])
    print(self.d.get_cls_counts())
    assert(np.all(self.d.get_cls_counts() == arr))

  def test_get_cls_ground_truth_json(self):
    table = Table(
      np.array([ [ True, True, False],
        [ True, False, False],
        [ False, True, False],
        [ False, False, True] ]), ["A","B","C"])
    assert(self.d.get_cls_ground_truth()==table)

  def test_det_ground_truth_for_class_json(self):
    gt = self.d.get_det_gt_for_class("A",with_diff=True,with_trun=True)
    arr = np.array(
      [[ 0.,  0.,  0.,  0.,  0., 0., 0, 0.],
       [ 1.,  1.,  1.,  0.,  0., 0., 0., 1.]])
    cols = ['x','y','w','h','cls_ind','diff','trun','img_ind']
    print(gt.arr)
    assert(np.all(gt.arr == arr))
    assert(gt.cols == cols)

    # no diff or trun
    gt = self.d.get_det_gt_for_class("A",with_diff=False,with_trun=False)
    arr = np.array(
      [[ 0.,  0.,  0.,  0.,  0., 0., 0, 0.],
       [ 1.,  1.,  1.,  0.,  0., 0., 0., 1.]])
    cols = ['x','y','w','h','cls_ind','diff','trun','img_ind']
    print(gt.arr)
    assert(np.all(gt.arr == arr))
    assert(gt.cols == cols)

  def test_set_class_values(self):
    assert(np.all(self.d.values == 1/3. * np.ones(len(self.classes))))
    self.d.set_class_values('uniform')
    assert(np.all(self.d.values == 1/3. * np.ones(len(self.classes))))
    self.d.set_class_values('inverse_prior')
    print(self.d.values)
    assert(np.all(self.d.values == np.array([0.25,0.25,0.5])))

class TestDatasetPascal:
  def setup(self):
    self.d = Dataset('test_pascal_train')
    
  def test_ground_truth_pascal_train(self):
    assert(self.d.num_classes() == 20)
    assert('dog' in self.d.classes)

  def test_ground_truth_for_class_pascal(self):
    correct = np.array(
      [[  48.,  240.,  148.,  132.,    11.,  0., 1., 0.]])
    ans = self.d.get_det_gt_for_class("dog")
    print ans
    assert np.all(ans.arr == correct)
      
  def test_neg_samples(self):
    # unlimited negative examples
    indices = self.d.get_neg_samples_for_class("dog",with_diff=True,with_trun=True)
    correct = np.array([1,2])
    assert(np.all(indices == correct))

    # maximum 1 negative example
    indices = self.d.get_neg_samples_for_class("dog",1,with_diff=True,with_trun=True)
    correct1 = np.array([1])
    correct2 = np.array([2])
    print(indices)
    assert(np.all(indices == correct1) or np.all(indices == correct2))

  def test_pos_samples(self):
    indices = self.d.get_pos_samples_for_class("dog")
    correct = np.array([0])
    assert(np.all(indices == correct))
    
  def test_ground_truth_test(self):
    d = Dataset('test_pascal_val')
    gt = d.get_det_gt(with_diff=False,with_trun=False)
    correct = np.matrix(
        [ [ 139.,  200.,   69.,  102.,   18.,   0., 0., 0.],
          [ 123.,  155.,   93.,   41.,   17.,   0., 0., 1.],
          [ 239.,  156.,   69.,   50.,    8.,   0., 0., 1.]])
    print(gt)
    assert np.all(gt.arr == correct)

  def test_get_pos_windows(self):
    d = Dataset('test_pascal_val')
    # TODO
    
  def test_kfold(self):
    """
    'sizes' here are empirical values over the trainval set.
    """
    d = Dataset('full_pascal_trainval')
    numfolds = 4
    d.create_folds(numfolds)
    cls = 'dog'
    sizes = [314, 308, 321, 320]
    for i in range(len(d.folds)):
      d.next_folds()
      pos = d.get_pos_samples_for_fold_class(cls)
      neg = d.get_neg_samples_for_fold_class(cls, pos.shape[0])
      assert(pos.shape[0] == sizes[i])
      assert(neg.shape[0] == sizes[i])

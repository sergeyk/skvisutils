from context import *
import time
from skpyutils import Table, skutil

from skvisutils import Image, BoundingBox, WindowParams

class TestImage(object):
  def setup(self):
    self.data = {
      "name": "test_image",
      "size": [640,480],
      "objects": [
        {"class":"A", "bbox": [0,0,0,0], "diff": 0, "trun": 0},
        {"class":"B", "bbox": [1,1,1,1], "diff": 0, "trun": 0},
        {"class":"C", "bbox": [2,2,2,2], "diff": 0, "trun": 0}
      ]
    }
    self.classes = ['A','B','C']
    self.columns = BoundingBox.columns+['cls_ind','diff','trun']

  def test_constructor(self):
    image = Image(20,10,['A','B','C'],'test_image')
    assert(image.width == 20 and image.height == 10)
    assert(image.classes == ['A','B','C'])
    assert(image.name == 'test_image')

  def test_get_whole_image_bbox(self):
    image = Image(20,10,[],'test_image')
    assert(image.get_whole_image_bbox() == BoundingBox((0,0,20,10)))
    image = Image(2,100,[],'test_image')
    assert(image.get_whole_image_bbox() == BoundingBox((0,0,2,100)))

  def test_load_json_data(self):
    image = Image.load_from_json(self.classes,self.data)
    assert(image.width == 640 and image.height == 480)
    assert(image.classes == ['A','B','C'])
    assert(image.name == 'test_image')
    objects_table = Table(np.array([
      [0,0,0,0,0,0,0],
      [1,1,1,1,1,0,0],
      [2,2,2,2,2,0,0]]), self.columns)
    assert(image.objects_table == objects_table)

  def test_get_det_gt(self):
    image = Image.load_from_json(self.classes,self.data)
    objects_table = Table(np.array([
      [0,0,0,0,0,0,0],
      [1,1,1,1,1,0,0],
      [2,2,2,2,2,0,0]]), self.columns)
    assert(image.get_objects() == objects_table)

    data = self.data.copy()
    data['objects'][0]['diff'] = 1
    data['objects'][1]['trun'] = 1
    image = Image.load_from_json(self.classes,data)
    objects_table = Table(np.array([
      [0,0,0,0,0,1,0],
      [1,1,1,1,1,0,1],
      [2,2,2,2,2,0,0]]), self.columns)
    assert(image.get_objects(with_diff=True,with_trun=True) == objects_table)

    objects_table = Table(np.array([
      [1,1,1,1,1,0,1],
      [2,2,2,2,2,0,0]]),self.columns)
    assert(image.get_objects(with_diff=False,with_trun=True) == objects_table)

    # this should be default behavior
    assert(image.get_objects() == objects_table)

    objects_table = Table(np.array([
      [2,2,2,2,2,0,0]]),self.columns)
    assert(image.get_objects(with_diff=False,with_trun=False) == objects_table)

    objects_table = Table(np.array([
      [0,0,0,0,0,1,0],
      [2,2,2,2,2,0,0]]), self.columns)
    assert(image.get_objects(with_diff=True,with_trun=False) == objects_table)

    # What if everything is filtered out?
    data['objects'] = data['objects'][:-1]
    objects_table = Table(np.array([
      [0,0,0,0,0,1,0],
      [1,1,1,1,1,0,1]]), self.columns)
    image = Image.load_from_json(self.classes,data)
    assert(image.get_objects(with_diff=True,with_trun=True) == objects_table)
    assert(image.get_objects(with_diff=False,with_trun=False).shape[0] == 0)

  def test_get_cls_counts_and_gt(self):
    data = self.data.copy()
    image = Image.load_from_json(self.classes,data)
    assert(np.all(image.get_cls_counts() == np.array([1,1,1])))
    assert(np.all(image.get_cls_gt() == np.array([True,True,True])))
    assert(image.contains_class('A') == True)
    assert(image.contains_class('B') == True)

    data['objects'][0]['class'] = 'B'
    image = Image.load_from_json(self.classes,data)
    # doesn't actually have to be Series, can be ndarray for comparison
    assert(np.all(image.get_cls_counts() == np.array([0,2,1])))
    assert(np.all(image.get_cls_gt() == np.array([False,True,True])))
    assert(image.contains_class('A') == False)
    assert(image.contains_class('B') == True)

    data['objects'] = []
    image = Image.load_from_json(self.classes,data)
    assert(np.all(image.get_cls_counts() == np.array([0,0,0])))
    assert(np.all(image.get_cls_gt() == np.array([False,False,False])))
    assert(image.contains_class('A') == False)
    assert(image.contains_class('B') == False)

  def test_get_random_windows(self):
    image = Image(width=3,height=2,classes=[],name='test')
    window_params = WindowParams(
        min_width=2,stride=1,scales=[1,0.5],aspect_ratios=[0.5])
    windows = image.get_random_windows(window_params,2)
    assert(windows.shape[0] == 2)
    windows = image.get_random_windows(window_params,3)
    assert(windows.shape[0] == 3)

  def test_get_windows_lots(self):
    t = time.time()
    image = Image(width=640,height=480,classes=[],name='test')
    window_params = WindowParams()
    window_params.min_width=10
    window_params.stride=8
    window_params.aspect_ratios=[0.5,1,1.5]
    window_params.scales=1./2**np.array([0,0.5,1,1.5,2])
    print(window_params)
    windows = image.get_windows(window_params)
    time_passed = time.time()-t
    print("Generating windows took %.3f seconds"%time_passed)
    print(np.shape(windows))
    print(windows[:10,:])
    rand_ind = np.random.permutation(np.shape(windows)[0])[:10]
    print(windows[rand_ind,:])

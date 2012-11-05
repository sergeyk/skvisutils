from synthetic.common_imports import *

from synthetic.image import *
from synthetic.sliding_windows import *
import synthetic.config as config

class TestSlidingWindows:
  def test_get_windows(self):
    image = Image(3,2,['whatever'],'test')
    window_params = WindowParams(
        min_width=2,stride=1,scales=[1,0.5],aspect_ratios=[0.5])
    windows = image.get_windows(window_params)
    correct = np.array(
        [[ 0., 0., 2., 2.],
         [ 0., 0., 3., 2.],
         [ 1., 0., 3., 2.],
         [ 2., 0., 2., 2.],
         [ 0., 1., 2., 2.],
         [ 0., 1., 3., 2.],
         [ 1., 1., 3., 2.],
         [ 2., 1., 2., 2.],
         [ 0., 0., 3., 3.],
         [ 0., 0., 4., 3.]])
    assert(windows.shape[0] > 0)
    print(windows)
    assert(np.all(correct == windows))
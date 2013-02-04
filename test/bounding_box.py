from context import *
from skvisutils.bounding_box import BoundingBox
from skpyutils.tictoc import TicToc


def test_convert_to_and_fro():
    bb = np.array([139., 200., 69., 102.])
    bb_c = BoundingBox.convert_arr_to_corners(bb)
    bb2 = BoundingBox.convert_arr_from_corners(bb_c)
    assert(np.all(bb == bb2))


def test_convert_to_and_fro_with_array():
    bb1 = np.array([139., 200., 69., 102.])
    bb2 = np.array([139., 200., 69., 51.])
    bb3 = np.array([239., 300., 69., 51.])
    bb = np.vstack((bb1, bb2, bb3))
    bb = np.tile(bb, (100000, 1))
    print(bb.shape)
    tt = TicToc().tic()
    bb_c = BoundingBox.convert_arr_to_corners(bb)
    tt.toc()
    print(bb_c.shape)
    tt.tic()
    bb2 = BoundingBox.convert_arr_from_corners(bb_c)
    tt.toc()
    print(bb2.shape)
    assert(np.all(bb == bb2))


def test_get_overlap():
    bbgt = np.array([139., 200., 69., 102.])
    bb = np.array([139., 200., 69., 102.])
    ov = BoundingBox.get_overlap(bb, bbgt)
    print(ov)
    assert(ov == 1)

    bb = np.array([139., 200., 69., 51.])
    ov = BoundingBox.get_overlap(bb, bbgt)
    print(ov)
    assert(ov == 0.5)

    bb = np.array([139., 200., 35., 51.])
    ov = BoundingBox.get_overlap(bb, bbgt)
    print(ov)
    assert((ov >= 0.24) and (ov <= 0.26))

    # switch order of arguments
    bb = np.array([139., 200., 35., 51.])
    ov = BoundingBox.get_overlap(bbgt, bb)
    print(ov)
    assert((ov >= 0.24) and (ov <= 0.26))

    bb = np.array([239., 300., 69., 51.])
    ov = BoundingBox.get_overlap(bb, bbgt)
    print(ov)
    assert(ov == 0)


def test_get_overlap_with_array():
    bbgt = np.array([139., 200., 69., 102.])
    bb1 = np.array([139., 200., 69., 102.])
    bb2 = np.array([139., 200., 69., 51.])
    bb3 = np.array([239., 300., 69., 51.])
    bb = np.vstack((bb1, bb2, bb3))
    numtimes = 100000
    bb = np.tile(bb, (numtimes, 1))
    tt = TicToc().tic()
    ov = BoundingBox.get_overlap(bb, bbgt)
    tt.toc()
    tt.tic()
    ov = BoundingBox.get_overlap(bb, bbgt)
    tt.toc()
    print(ov)
    assert(np.all(ov == np.tile(np.array([1, 0.5, 0]), numtimes)))

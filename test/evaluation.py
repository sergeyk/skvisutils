from context import *
import glob

from skpyutils import Table
from skvisutils import evaluation, Dataset


class TestEvaluationSynthetic(unittest.TestCase):
    def setUp(self):
        self.d = Dataset(test_config, 'test_data2')
        self.d.load_from_json(test_data2)
        self.classes = self.d.classes

        det_gt = self.d.get_det_gt()

        # perfect detections
        scores = np.ones(det_gt.shape[0])
        self.full_dets = det_gt.append_column('score', scores)

        # perfect detections, but only for class 'A'
        dets_just_A = self.d.get_det_gt_for_class('A')
        scores = np.ones(dets_just_A.shape[0])
        self.partial_dets = dets_just_A.append_column('score', scores)

    def test_values(self):
        det_gt = self.d.get_det_gt()

        self.d.set_class_values('uniform')
        assert(np.all(self.d.values == 1. / 3 * np.ones(len(self.classes))))

        ap = evaluation.compute_det_map(self.full_dets, det_gt, self.d.values)
        assert(ap == 1)
        ap = evaluation.compute_det_map(
            self.partial_dets, det_gt, self.d.values)
        assert_almost_equal(ap, 1 / 3.)

        self.d.set_class_values('inverse_prior')
        assert(np.all(self.d.values == np.array([0.25, 0.25, 0.5])))

        ap = evaluation.compute_det_map(self.full_dets, det_gt, self.d.values)
        assert(ap == 1)
        ap = evaluation.compute_det_map(
            self.partial_dets, det_gt, self.d.values)
        assert_almost_equal(ap, 0.25)

    def test_compute_pr_multiclass(self):
        cols = ['x', 'y', 'w', 'h', 'cls_ind', 'img_ind', 'diff']
        dets_cols = ['x', 'y', 'w', 'h', 'score', 'time', 'cls_ind', 'img_ind']

        # two objects of different classes in the image, perfect detection
        arr = np.array(
            [[0, 0, 10, 10, 0, 0, 0],
             [10, 10, 10, 10, 1, 0, 0]])
        gt = Table(arr, cols)

        dets_arr = np.array(
            [[0, 0, 10, 10, -1, -1, 0, 0],
             [10, 10, 10, 10, -1, -1, 1, 0]])
        dets = Table(dets_arr, dets_cols)

        # make sure gt and gt_cols aren't modified
        gt_arr_copy = gt.arr.copy()
        gt_cols_copy = list(gt.cols)
        ap, rec, prec = evaluation.compute_det_pr(dets, gt)
        assert(np.all(gt.arr == gt_arr_copy))
        assert(gt_cols_copy == gt.cols)

        correct_ap = 1
        correct_rec = np.array([0.5, 1])
        correct_prec = np.array([1, 1])
        print((ap, rec, prec))
        assert(correct_ap == ap)
        assert(np.all(correct_rec == rec))
        assert(np.all(correct_prec == prec))

        # some extra detections to generate false positives
        dets_arr = np.array(
            [[0, 0, 10, 10, -1, -1, 0, 0],
             [0, 0, 10, 10, 0, -1, 0, 0],
             [10, 10, 10, 10, 0, -1, 1, 0],
             [10, 10, 10, 10, -1, -1, 1, 0]])
        dets = Table(dets_arr, dets_cols)

        ap, rec, prec = evaluation.compute_det_pr(dets, gt)
        correct_rec = np.array([0.5, 1, 1, 1])
        correct_prec = np.array([1, 1, 2. / 3, 0.5])
        print((ap, rec, prec))
        assert(np.all(correct_rec == rec))
        assert(np.all(correct_prec == prec))

        # confirm that running on the same dets gives the same answer
        ap, rec, prec = evaluation.compute_det_pr(dets, gt)
        correct_rec = np.array([0.5, 1, 1, 1])
        correct_prec = np.array([1, 1, 2. / 3, 0.5])
        print((ap, rec, prec))
        assert(np.all(correct_rec == rec))
        assert(np.all(correct_prec == prec))

        # now let's add two objects of a different class to gt to lower recall
        arr = np.array(
            [[0, 0, 10, 10, 0, 0, 0],
             [10, 10, 10, 10, 1, 0, 0],
             [20, 20, 10, 10, 2, 0, 0],
             [30, 30, 10, 10, 2, 0, 0]])
        gt = Table(arr, cols)
        ap, rec, prec = evaluation.compute_det_pr(dets, gt)
        correct_rec = np.array([0.25, 0.5, 0.5, 0.5])
        correct_prec = np.array([1, 1, 2. / 3, 0.5])
        print((ap, rec, prec))
        assert(np.all(correct_rec == rec))
        assert(np.all(correct_prec == prec))

        # now call it with empty detections
        dets_arr = np.array([])
        dets = Table(dets_arr, dets_cols)
        ap, rec, prec = evaluation.compute_det_pr(dets, gt)
        correct_ap = 0
        correct_rec = np.array([0])
        correct_prec = np.array([0])
        print((ap, rec, prec))
        assert(np.all(correct_ap == ap))
        assert(np.all(correct_rec == rec))
        assert(np.all(correct_prec == prec))

    def test_plots(self):
        full_results_dirname = os.path.join(res_dir, 'full_dets_eval')
        partial_results_dirname = os.path.join(res_dir, 'partial_dets_eval')

        evaluation.evaluate_detections_whole(
            self.d, self.partial_dets, partial_results_dirname, force=True)
        assert(os.path.exists(
            os.path.join(partial_results_dirname, 'whole_dashboard.html')))
        pngs = glob.glob(os.path.join(partial_results_dirname, '*.png'))
        assert(len(pngs) == 4)  # 3 classes + 1 multiclass

        evaluation.evaluate_detections_whole(
            self.d, self.full_dets, full_results_dirname, force=True)

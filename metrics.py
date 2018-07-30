import numpy as np
import cv2 as cv
from scipy.spatial.distance import euclidean as l2

DEBUG = False # turn off debug for faster calculation
RATIO = 1.


def set_debug(debug):
    global DEBUG
    DEBUG = debug

def get_debug():
    return DEBUG

def draw_box(img, boxes, cls):
    img = np.array(img).copy()
    for b in boxes:
        cv.fillConvexPoly(img, b, cls)
    return img


def mean_iou(labels, predictions, num_classes, weights):
    """
    :param labels: ground truth map - each pixel belongs to 1 of 2 classes [0, num_classes-1]
    :param predictions: same as labels
    :param num_classes:
    :param weights: weight for each class to calculate the mean iou - dim (num_classes-1, )
    :return: mean iou and iou of each class
    """

    assert type(labels) in [list, np.ndarray], 'Labels must be a list or np array'
    assert type(predictions) in [list, np.ndarray], 'Prediction must be a list or np array'

    labels = np.array(labels).flatten()
    predictions = np.array(predictions).flatten()
    weights = np.array(weights).flatten()

    assert labels.shape == predictions.shape, 'Shapes of labels and predictions must be the same'
    assert weights.shape[0] == num_classes

    ans = []
    for cls in range(num_classes):
        lab = labels == cls
        pre = predictions == cls

        inter = lab&pre
        union = lab|pre
        ans.append(inter.sum()/union.sum())

    return np.mean(np.array(ans)*weights)


class base_error:
    def __init__(self):

        self.gt_map = None
        self.pr_map = None

    def __get_shape(self, boxes):
        return boxes.reshape(-1,2).max(0)

    def __get_patch(self, box):
        w = int(max(l2(box[0], box[1]), l2(box[2], box[3]))) + 1
        h = int(max(l2(box[3], box[0]), l2(box[1], box[2]))) + 1
        # print('Box size', w, h)

        poly = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]).astype('float32')
        matrix = cv.getPerspectiveTransform(box.astype('float32'), poly)
        patch = cv.warpPerspective(self.pr_map, matrix, (w, h))

        # DEBUG
        db_gt = db_pr = None
        if DEBUG:
            db_gt = cv.warpPerspective(self.gt_map, matrix, (w, h))
            tmp = np.stack([(self.pr_map>0).astype(np.uint8)*255, self.img, self.img], 2)
            db_pr = cv.warpPerspective(tmp, matrix, (w, h))

        # remove unintended edge of other regions
        pad = 5
        patch[:pad, :] = patch[-pad:, :] = patch[:, :pad] = patch[:, -pad:] = 0
        if DEBUG:
            db_gt[:pad, :] = db_gt[-pad:, :] = db_gt[:, :pad] = db_gt[:, -pad:] = 0
            db_pr[:pad, :] = db_pr[-pad:, :] = db_pr[:, :pad] = db_pr[:, -pad:] = 0
        return patch, db_gt, db_pr

    # @staticmethod
    def __cal_err(self, gt, pred, img=None):
        gt = np.array(gt)
        pred = np.array(pred)
        # TODO: sort predicted boxes by its area to prevent bigger boxes from covering the small ones.
        assert len(gt.shape) == len(pred.shape) == 3
        # return 0
        shape = img.shape if img is not None else None
        self.img = img
        if shape is None:
            b1 = self.__get_shape(gt)
            b2 = self.__get_shape(pred)
            shape = (max(b1[0], b2[0])+1, max(b1[1], b2[1])+1)

            self.img = np.zeros(shape)

        self.pr_map = np.zeros_like(self.img)
        for idx, b in enumerate(pred):
            cv.fillConvexPoly(self.pr_map, b, idx+1)
        # self.pr_map = self.pr_map*0.3 + imgs*0.7
        # ===================DEBUG===================
        self.gt_map = np.zeros_like(self.img)
        # self.gt_map = self.gt_map*0.3 + imgs*0.7
        for idx, b in enumerate(gt):
            cv.fillConvexPoly(self.gt_map, b, idx + 1)
        # ===========================================
        errs = []
        db_list = []
        for idx, b in enumerate(gt):
            patch, gt_patch, pr_patch = self.__get_patch(b)
            # patch =

            # TODO: checking if each small box is meaningful instead of just considering its area.
            out = cv.connectedComponentsWithStats((patch>0).astype(np.uint8))
            n_regions, label_matrix, stats, centroids = out[0], out[1], out[2], out[3]
            n_active_region = n_regions-1 # ignore background
            errs.append(n_active_region)
            if n_active_region > 1:
                db_list.append(pr_patch)

        errs = np.array(errs)-1
        errs = (errs>0)*errs
        ret = {'Total error': errs.sum(),
               'Mean error': errs.mean(),
               'STD': errs.std()}
        # print(ret, db_list)

        return ret, db_list


class split_error(base_error):
    def __init__(self):
        super(split_error, self).__init__()

    def __call__(self, gt, pred, img = None):
        """
        :param gt: array of bounding boxes (quadrilateral) in groundtruth
        :param pred: same as gt
        :param img: original document - used for debug
        :return: total splits, average split per line, std split
        """
        if img is not None:
            img = cv.resize(img, (0, 0), fx=RATIO, fy=RATIO)
        gt = (np.array(gt) * RATIO).astype(np.int32)
        pred = (np.array(pred) * RATIO).astype(np.int32)
        return self._base_error__cal_err(gt, pred, img)


class merge_error(base_error):
    def __init__(self):
        super(merge_error, self).__init__()

    def __call__(self, gt, pred, img=None):
        if img is not None:
            img = cv.resize(img, (0, 0), fx=RATIO, fy=RATIO)

        gt, pred = pred, gt

        gt = (np.array(gt) * RATIO).astype(np.int32)
        pred = (np.array(pred) * RATIO).astype(np.int32)
        return self._base_error__cal_err(gt, pred, img)
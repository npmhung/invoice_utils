from keras.models import model_from_json
import cv2
import numpy as np
from scipy.spatial import distance as dist
from math import sin, cos, radians, pi
import os


def mid_point(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return ((x1 + x2)/2, (y1 + y2)/2)


def mid_line(pts):
    [tl, tr, br, bl] = order_points(pts)
    ave_height = (dist.euclidean(tl,bl) + dist.euclidean(tr,br)) / 2
    return mid_point(tl,bl),mid_point(tr,br),ave_height


def point_pos(x0, y0, d, theta):
    """
    get a point position distance d , angle theta from point x0,y0
    """
    theta_rad = radians(theta)
    return x0 + d*cos(theta_rad), y0 + d*sin(theta_rad)


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def cal_box(x0, y0, x1, y1, height):
    """
    Cal box with height from line x0,y0,x1,y1
    """
    padding = 1
    angle = np.rad2deg(np.arctan2(y1 - y0, x1 - x0))
#    factor = angle/abs(angle)
    x0, y0 = point_pos(x0, y0, -1*padding,angle)
    x2, y2 = point_pos(x0, y0, height/2.0,angle-90)
    x3, y3 = point_pos(x0, y0, height/2.0,angle+90)
    x1, y1 = point_pos(x1, y1, padding, angle)
    x4, y4 = point_pos(x1, y1, height/2.0,angle+90)
    x5, y5 = point_pos(x1, y1, height/2.0,angle-90)
    pts = np.asarray([[x2, y2], [x3, y3], [x4, y4], [x5, y5]])
    box = order_points(pts)
    return box


class LineCutModel:
    def __init__(self):
        self.model = None
        self.img = None

    def load(self, **config):
        self.model = model_from_json(open(config['model_json'], 'r').read())
        self.model.load_weights(config['model_weight'])
        pass

    def __load_image(self, img_file):
        if isinstance(img_file, np.ndarray):
            self.img = img_file
        elif isinstance(img_file, str):
            if len(img_file) > 0:
                try:
                    img = cv2.imread(img_file)
                    self.img = img
                except Exception as e:
                    raise ValueError('Error - Error reading file')
            else:
                raise ValueError('Error - Empty image path')

        if len(self.img.shape) > 2:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        return self.img

    def __process(self, img):
        """
        Run prediction
        """
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, -1)
        if np.amax(img) > 1.0:
            img = img / 255
        line = self.model.predict(img, batch_size=1, verbose=1)
        line = (line > 0.9).astype(np.uint8)
        return line

    def __preprocess(self, img):
        h, w = img.shape
        min_size = min(w, h)
        max_size = max(w, h)

        factor = max_size / 1024
        if factor > 1.0:
            w = int(w/factor)
            h = int(h/factor)
        w = w // 32 * 32
        h = h // 32 * 32

        return cv2.resize(img, (w, h))

    def __postprocess(self, originImage, textLineImage, scale=1.667, mode=2):
        """
        Group box and return result
        """
        height, width = originImage.shape
        debug_im = np.ones((height, width, 3), np.uint8) * 255
        debug_im[:, :, 0] = originImage
        debug_im[:, :, 1] = originImage
        debug_im[:, :, 2] = originImage
        overlay = debug_im.copy()
        output = debug_im.copy()
        grayImage = textLineImage
        _, thresh = cv2.threshold(grayImage, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        it = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if mode == 1:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), -1)
                y = int(y - h * 0.25)
                h = int(h * 1.5)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 255), 1)
                boxes.append([x, y, x + w, y + h])
            elif mode == 2:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # print("boxb: ",box)
                # print("boxf: ",mid_line(box))
                if 10 <= mid_line(box)[2] <= 20:
                    color = (255, 0, 0)
                elif  20 < mid_line(box)[2] <= 30:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                if 3 < mid_line(box)[2] < 100:
                    cv2.drawContours(overlay, [np.int0(box)], 0, color, -1)
                    x0 = mid_line(box)[0][0]
                    y0 = mid_line(box)[0][1]
                    x1 = mid_line(box)[1][0]
                    y1 = mid_line(box)[1][1]
                    h = mid_line(box)[2] * scale
                    box_r = cal_box(x0, y0, x1, y1, h)
                    # box_r = np.int0(box_r)
                    # print("box r:",box_r)
                    cv2.drawContours(overlay, [np.int0(box_r)], 0, color, 1)
                    # cv2.line(overlay,mid_line(box)[0],mid_line(box)[1],color,int(mid_line(box)[2] * 1.15))
                    boxes.append(box_r)
            elif mode == 3:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                # box_r = cal_box(x0,y0,x1,y1,h)
                # box_r = np.int0(box_r)
                cv2.drawContours(overlay, [np.int0(box)], 0, (0, 0, 0), -1)
            it = it + 1
            # textLine.sort(key=lambda x: x[1]+x[3])

        # apply the overlay
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return boxes, output

    def __extract_line(self, im, boxes):
        def l2(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        lines = []
        for box in boxes:
            w = int(max(l2(box[0], box[1]), l2(box[2], box[3]))) + 1
            h = int(max(l2(box[3], box[0]), l2(box[1], box[2]))) + 1
            # print('Box size', w, h)

            poly = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]).astype('float32')
            matrix = cv2.getPerspectiveTransform(box.astype('float32'), poly)
            lines.append(cv2.warpPerspective(im, matrix, (w, h)))
        return lines

    def __text_line_segmentation(self, image, scale=1.667, mode=2):
        """
        scale = 1.65~1.667 the box is 60% percentage of the height
        mode = 1 return 2 points , 2 return 4 points
        """
        h_in, w_in = image.shape
        img = self.__preprocess(image)
        h_out, w_out = img.shape
        w_scale = w_in / w_out
        h_scale = h_in / h_out
        # print(w_scale, h_scale)

        line = self.__process(img)

        boxes, debug_im = self.__postprocess(img, line[0, :, :, 1] * 255, scale=scale, mode=mode)
        # recover original
        for pts in boxes:
            if type(pts) is not list:
                for pt in pts:
                    pt[0] = int(pt[0] * w_scale)
                    pt[1] = int(pt[1] * h_scale)
            else:
                pts[0] = int(pts[0] * w_scale)
                pts[2] = int(pts[2] * w_scale)
                pts[1] = int(pts[1] * h_scale)
                pts[3] = int(pts[3] * h_scale)
        # extract straight line
        boxes = sorted(boxes, key=lambda box: (box[0][1], box[0][1]))
        lines = self.__extract_line(np.array(image), boxes)
        return boxes, lines, debug_im

    def predict(self, img_file, return_csv=False):
        """

        :param img_file: np array or path to the image
        :param return_csv: update later
        :return:
            - boxes: list of boxes [ [x1, y1, x2, y2, x3, y3, x4, y4], [x1, y1, x2, y2, x3, y3, x4, y4], ...]
            Each box only has 4 points and covers/ represents a line.

            - lines: including list of images. Each image is a line extracted from the document and applied some basic
            deskew/ dewarp operation (perspective transform)

            - debug_im: debug image -> visualize lines detected by the model.
        """
        assert self.model is not None, 'Model isn\'t loaded yet.'
        img = self.__load_image(img_file)
        boxes, lines, debug_im = self.__text_line_segmentation(img)
        flatten_boxes = [np.array(b).flatten('C').tolist() for b in boxes]
        return flatten_boxes, lines, debug_im

    def save(self, path):
        assert self.model is not None, 'Model isn\'t loaded/fitted yet.'
        try:
            os.makedirs(path)
        except:
            pass

        mpath = os.path.join(path, 'model')

        with open(mpath+'.json', 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(mpath+'.h5')

    def fit(self, X, y):
        """
        Fitting this model requires some augmentation operations, hence isn't suitable to implement here.
        """
        # print('Not implemented yet')
        pass

def main():
    def draw_debug(img, boxes):
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(img).astype('int32')
        for b in boxes:
            # print(b)
            cv2.fillConvexPoly(mask, np.array(b).reshape(-1, 2).astype('int32'), 255)

        return img*0.7+mask*0.3

    model = LineCutModel()
    model.load(model_json='model_final.json', model_weight='model_final.h5')

    # # input is np array
    tim = cv2.imread('SG1-9-1.jpg')
    b, lines, dim = model.predict(tim)
    cv2.imwrite('debug_1.png', dim)
    cv2.imwrite('debug_fullsize_1.png',draw_debug(tim, b))

    # input is a path
    tim = 'SG1-13.jpg'
    _tim = cv2.imread(tim)
    _tim = cv2.cvtColor(_tim, cv2.COLOR_BGR2GRAY)
    b, lines, dim = model.predict(tim)
    cv2.imwrite('debug_2.png', dim)
    cv2.imwrite('debug_fullsize_2.png', draw_debug(_tim, b))

    model.save('./my_model')
    print(b)

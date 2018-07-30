from utils.img_tools import *
from utils.linecut_model import LineCutModel

def paragraphs(boxes, shape):
    full_mask = draw_mask(boxes, shape)
    tmp = cv.dilate(full_mask, np.ones((5,1)), iterations=3)
    conts = cv.findContours(tmp, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE )
    return conts[1]
    
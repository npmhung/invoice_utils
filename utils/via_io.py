import pandas as pd
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs ,json
import os,cv2
import re

XML_EXT = '.xml'
CSV_EXT = '.csv'
ENCODE_METHOD = 'utf-8'

class VIAWriter:
    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False
        self.via_data = []
        self.via_data.append(["#filename", "file_size", "file_attributes", "region_count", "region_id", "region_shape_attributes", "region_attributes"])
        self.box_count = 0
        self.total = 999

    def addPolyBox(self,box):
        pre_str = "\"{\"\"name\"\":\"\"polygon\"\",\"\"all_points_x\"\""
        via_row = []
        via_row.append(self.filename)
        via_row.append(self.imgSize)
        via_row.append("\"{}\"")
        via_row.append(str(self.total))
        via_row.append(str(self.box_count))
        self.box_count += 1
        rec_code = box
        x1y1, x2y2, x3y3, x4y4 = rec_code.split(",")
        x1y1 = x1y1[1:-1]
        x2y2 = x2y2[1:-1]
        x3y3 = x3y3[1:-1]
        x4y4 = x4y4[1:-1]

        x1, y1 = x1y1.split()
        x2, y2 = x2y2.split()
        x3, y3 = x3y3.split()
        x4, y4 = x4y4.split()
        x5, y5 = x1, y1

        x_ss = "[{},{},{},{},{}]".format(str(x1), str(x2), str(x3), str(x4), str(x5))
        y_ss = "[{},{},{},{},{}]".format(str(y1), str(y2), str(y3), str(y4), str(y5))

        x_ss = x_ss.replace(".", "")
        y_ss = y_ss.replace(".", "")

        reg_text = "{}:{},\"\"all_points_y\"\":{}".format(pre_str, x_ss, y_ss)
        reg_text += "}\""
        via_row.append(reg_text)
        via_row.append("\"{}\"")
        self.via_data.append(via_row)

    def save(self, targetFile=None):
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)
        with out_file:
            for row in self.via_data:
                ss = str(row)
                ss = ss.replace("'", "")
                ss = ss.replace(" ", "")
                ss = ss[1:-1]
                out_file.write("{}\r\n".format(ss))
        out_file.close()
class VIAReader:
    """
    Reader class
    """
    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.boxes= []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseCSV()
        except:
            pass

    def parseCSV(self):
        assert self.filepath.endswith(CSV_EXT), "Unsupport file format"
        df = pd.read_csv(self.filepath)
        for index, row in df.iterrows():
            data = row['region_shape_attributes']
            data = re.sub(r"\.","",data)
            data = json.loads(data)
            box_type = data['name']
            box = []
            if box_type == 'polygon':
                for i in range(len(data['all_points_x'])-1): # subtract last point
                    box.append([data['all_points_x'][i],data['all_points_y'][i]])
                self.addPolyBox(box)
            elif box_type == 'rect':
                w = data['width']
                h = data['height']
                box.append([data['x']  ,data['y']  ])
                box.append([data['x']+w,data['y']  ])
                box.append([data['x']+w,data['y']+h])
                box.append([data['x']  ,data['y']+h])
                self.addRectBox(box)
            else:
                try:
                    raise ValueError
                except ValueError:
                    print("VIA_IO: Box Type invalid")
        return

    def addPolyBox(self,box):
        self.boxes.append(box)
        return True

    def addRectBox(self,box):
        self.boxes.append(box)
        return True

    def getBoxes(self):
        return self.boxes
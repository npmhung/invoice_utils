import xml.etree.ElementTree as et
import json
import numpy as np

def parse_coor_str(str_coords):
    return [*map(int, str_coords.replace(' ', ',').split(','))]

def athelia_parser(xml_path):
    """
    Args:
        xml_path:
    Returns:
        Dictionary {type1:[box1, box2, ..., boxn], type2:[box1, ..., boxm]}.
        box -> [x1, y1, ..., xk, yk]
    """
    tree = et.parse(xml_path)
    # it = ET.iterparse(StringIO(tree))
    root = tree.getroot()
    for elem in root.getiterator():
        if not hasattr(elem.tag, 'find'): continue  # (1)
        i = elem.tag.find('}')
        if i >= 0:
            elem.tag = elem.tag[i+1:]

    objs = dict()

    for region in root.find('Page').getchildren():
        t1 = region.tag
        t2 = region.attrib.get('type', '')

        if t2=='':
            _type=t1
        else:
            _type=t1+'_'+t2

        coords = parse_coor_str(region.find('Coords').attrib['points'])
        l = objs.get(_type, [])
        l.append(coords)
        objs[_type] = l
    
    return objs

def via_json_parser(path):
    objs = json.load(open(path, encoding='utf-8'))
    try:
        objs = objs['_via_img_metadata']
    except:
        pass
    
    ret = dict()
    for img in objs.values():
        boxes = []
        values = []
        regions = img['regions']
        for region in regions:
            shape_att = region['shape_attributes']
            if shape_att['name']=='polygon':
                xs = shape_att['all_points_x']
                ys = shape_att['all_points_y']
                box = np.array([xs,ys]).T.flatten()
                boxes.append(box)
            elif shape_att['name']=='rect':
                x=shape_att['x']
                y=shape_att['y']
                w=shape_att['width']
                h=shape_att['height']
                boxes.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
#             print(region['region_attributes'])
            values.append(region['region_attributes'].get('value', ''))
        ret[img['filename']]={'boxes':{'lines':boxes}, 'values':values}
    return ret
            
    
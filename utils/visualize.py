import os
import shutil
from utils.img_tools import *

import bokeh
from bokeh.plotting import figure, output_file
from bokeh.models import HoverTool, CheckboxGroup, CustomJS
import bokeh.palettes as bp
from bokeh.resources import CDN
from bokeh.embed import file_html


def doccoor2planecoor(document_boxes_y, document_height):
    """
    Convert coordinates on document to corresponding coordinates on Cartesian plane.
    Args:
        document_boxes: [[y1, ..., yn], [y1, ... ym], ...]
    """
    
    ret = [document_height-np.array(y)-1 for y in document_boxes_y]
    return ret


def render_mask(img_path, document_objs, document_name='Default_Document', output_path='./outputs/', save_html=False):
    img = load_img(img_path)
    bokeh.plotting.reset_output()
    hover = HoverTool(
        tooltips=[
            ("Name", "@name"),
            ("(x,y)", "($x, $y)"),
        ]
    )
    fig = figure(tools=[hover,
                        bokeh.models.WheelZoomTool(),
                        bokeh.models.PanTool()])
    fig.hover.point_policy = "follow_mouse"
    fig.sizing_mode='scale_width'
    fig.image_url(url=[os.path.basename(img_path)], x=0, y=img.shape[0], w=img.shape[1], h=img.shape[0])
    
    script = "active = cb_obj.active;"
    labels = list(document_objs.keys())
    color_map = bp.magma(len(labels))
    
    args = dict()
    total_objs = 0
    for key_id, key in enumerate(labels):
        xs = []
        ys = []
        for box in document_objs[key]:
            _ = np.array(box).reshape(-1,2).T
            xs.append(_[0])
            ys.append(_[1])
        ys=doccoor2planecoor(ys, img.shape[0])
        data_source = dict(x=xs, y=ys, 
                          name=["%s %d"%(key, idx) for idx in range(len(xs))])
        
        falpha = 0.5*int('table' not in key.lower())
        lcolor = 'blue' if 'table' in key.lower() else 'red'
        lwidth = 3 if 'table' in key.lower() else 1
        total_objs += len(document_objs[key])
        args[key] = fig.patches('x', 'y', source=data_source, fill_color=color_map[key_id], 
                                fill_alpha=falpha, line_color=lcolor, line_width=lwidth, 
                                legend=key+': %d'%len(document_objs[key]))
        r = args[key]
        script += "\n%s.visible = active.includes(%d);"%(key, key_id)
    fig.patch([],[], fill_color='red', fill_alpha=0, line_color='white', legend='Total region: %d'%total_objs)
    fig.legend.click_policy="hide"
    fig.legend.location = "top_left"
    checkbox = CheckboxGroup(labels=labels, active=[])#active=[*range(len(labels))])
    checkbox.callback = CustomJS(args=args, code=script)
    plt_obj = [fig]

    
    html = file_html(plt_obj, CDN, title=document_name)
    if save_html:
        base = os.path.join(output_path, document_name)
        if os.path.exists(base):
            shutil.rmtree(base)
        os.makedirs(base)
        shutil.copy(img_path, base)
        with open(os.path.join(base, 'main.html'),'w') as g:
            g.write(html)
            
    return html
    
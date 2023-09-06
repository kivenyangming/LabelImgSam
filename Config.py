import time
import cv2
# python scripts/export_onnx_model.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./mobile_sam.onnx
from sortedcontainers import SortedDict

from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry


def load_mask():
    model_type = "vit_t"
    sam_checkpoint = "./weights/mobile_sam.pt"
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device="cuda")
    mobile_sam.eval()
    mask_generator = SamAutomaticMaskGenerator(mobile_sam)
    return mask_generator

def mask_infer(masks):
    area_list = [mask['area'] for mask in masks]  # 获取推理结果中的坐标框的面积集合
    xywh_list = [masks[i]['bbox'] for i in range(len(masks))]  # 获取推理结果中的bbox坐标集 此时是xywh
    xywh_list = [b for _, b in sorted(zip(area_list, xywh_list))]  # 根据面积由小到大对xywh排序
    return xywh_list

def xywh2xyxy(xywhs):
    return [[x, y, x + w, y + h] for x, y, w, h in xywhs]

def build_box_tree(xyxy_list):
    box_tree = SortedDict()
    for i, box in enumerate(xyxy_list):
        box_tree[i] = box
    return box_tree

def is_point_in_box(point, box):
    xmin, ymin, xmax, ymax = box
    x, y = point
    return xmin <= x <= xmax and ymin <= y <= ymax

def is_box_containing_point(box, point):
    return is_point_in_box(point[0], box) and is_point_in_box(point[1], box)

def load_list_tree(uuid):
    import ast
    xyxy_list = []
    with open('./saveInfo/xyxy_list_%s.txt'%uuid,  'r') as f:
        for line in f:
            info = line.strip()
            info = ast.literal_eval(info)
            xyxy_list.append(info)
    xyxy_tree = []
    with open('./saveInfo/xyxy_list_%s.txt'%uuid,  'r') as f:
        for line in f:
            info = line.strip()
            info = ast.literal_eval(info)
            xyxy_tree.append(info)
    return xyxy_list, xyxy_tree

def writ_base(xml_file, xml_name, width, height):

    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + str(xml_name) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

def writr_xyxy(xml_file, bbox):
    for box in bbox:
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str("car") + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

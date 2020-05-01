#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 19-10-21 下午2:49
# @Author  : jaykky
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os
import json


def get_dataset_xmls(annot_folder, dataset='trainval'):
    txt_file = os.path.abspath(os.path.join(annot_folder, '..', 'ImageSets', 'Main', '{}.txt'.format(dataset)))
    assert os.path.exists(txt_file)

    xml_list = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('\n', '')
            if line != '':
                xml_list.append(os.path.join(annot_folder, '{}.xml'.format(line)))
    return xml_list

def readtxt(class_file):
    classes = []
    with open(class_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('\n', '')
            if line!='':
                classes.append(line)
    return classes

def build_categories(classes):
    '''
    构建 类别信息
    :param classes:
    :return:
    '''
    category_items = []
    for category_item_id, category_item_name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = category_item_id + 1
        category_item['name'] = category_item_name
        category_items.append(category_item)
    return category_items

def getImgItem(file_name, size, image_id):
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')

    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']

    return image_item

def getAnnoItem(object_name, image_id, annotation_id, category_id, bbox):
    # global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    # annotation_id += 1
    annotation_item['id'] = annotation_id
    # coco['annotations'].append(annotation_item)
    return annotation_item

def build_cocos(coco, xml_list):
    for index, xml in enumerate(xml_list):
        if not xml.endswith('.xml'):
            continue

        coco = build_coco(coco, xml, index)
    return coco

def build_coco(coco, xml, index):
    image_id = 20180000000 + index
    annotation_id = 0 + index

    bndbox = dict()
    size = dict()
    current_image_id = None
    current_category_id = None
    file_name = None
    size['width'] = None
    size['height'] = None
    size['depth'] = None

    category_set = dict()
    for item in coco['categories']:
        category_set[item['name']] = item['id']

    tree = ET.parse(xml)
    root = tree.getroot()
    if root.tag!='annotation':
        raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

    # elem is <folder>, <filename>, <size>, <object>

    for elem in root:
        current_parent = elem.tag
        current_sub = None
        object_name = None

        if elem.tag=='folder':
            continue

        if elem.tag=='filename':
            file_name = elem.text
            # if file_name in category_set:
            #     raise Exception('file_name duplicated')

        # add img item only after parse <size> tag
        elif current_image_id is None and file_name is not None and size['width'] is not None:
            # if file_name not in image_set:
            #     current_image_id = addImgItem(file_name, size)
            #     print('add image with {} and {}'.format(file_name, size))
            # else:
            #     raise Exception('duplicated image: {}'.format(file_name))
                # subelem is <width>, <height>, <depth>, <name>, <bndbox>

            current_image_id = image_id
            coco['images'].append(getImgItem(file_name, size, current_image_id))
            print('add image with {} and {}'.format(file_name, size))

        for subelem in elem:
            bndbox['xmin'] = None
            bndbox['xmax'] = None
            bndbox['ymin'] = None
            bndbox['ymax'] = None

            current_sub = subelem.tag
            if current_parent=='object' and subelem.tag=='name':
                object_name = subelem.text
                if object_name not in category_set.keys():
                    #
                    print(object_name, 'not in target label')
                    # break

                else:
                    current_category_id = category_set[object_name]

            elif current_parent=='size':
                if size[subelem.tag] is not None:
                    raise Exception('xml structure broken at size tag.')
                size[subelem.tag] = int(subelem.text)

            # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
            for option in subelem:
                if current_sub=='bndbox':
                    if bndbox[option.tag] is not None:
                        raise Exception('xml structure corrupted at bndbox tag.')
                    bndbox[option.tag] = int(option.text)

            # only after parse the <object> tag
            if bndbox['xmin'] is not None:
                if object_name is None:
                    raise Exception('xml structure broken at bndbox tag')
                if current_image_id is None:
                    raise Exception('xml structure broken at bndbox tag')
                if current_category_id is None:
                    raise Exception('xml structure broken at bndbox tag')
                bbox = []
                # x
                bbox.append(bndbox['xmin'])
                # y
                bbox.append(bndbox['ymin'])
                # w
                bbox.append(bndbox['xmax'] - bndbox['xmin'])
                # h
                bbox.append(bndbox['ymax'] - bndbox['ymin'])
                # print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                #                                                bbox))
                # addAnnoItem(object_name, current_image_id, current_category_id, bbox)
                coco['annotations'].append(getAnnoItem(object_name, current_image_id, annotation_id,
                                                       current_category_id, bbox))
    return coco



def get_coco_json(annot_folder, save_folder, class_file, dataset='trainval'):
    # 一个线索，贯穿全局
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []

    # 确定目标类别
    classes = readtxt(class_file)

    # 锁定目标类别结构
    coco['categories'] = build_categories(classes)

    # 确定文件队列
    xml_list = get_dataset_xmls(annot_folder, dataset)
    # 根据文件内容，构建内容结构
    coco = build_cocos(coco, xml_list)

    # 保存
    json_file = os.path.join(save_folder, '{}.json'.format(dataset))  # 这是你要生成的json文件
    json.dump(coco, open(json_file, 'w'))

if __name__=='__main__':
    annot_folder = '/data1/syh/ice_locker/total_data/Annotations'
    class_file = './classes.txt'
    save_folder = os.path.abspath(os.path.join(annot_folder, '..', 'annotations_coco'))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    get_coco_json(annot_folder, save_folder, class_file, dataset='trainval')
    get_coco_json(annot_folder, save_folder, class_file, dataset='test')
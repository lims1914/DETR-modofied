import os
import json
import numpy as np
import xml.etree.ElementTree as ET

## code annotation and image into json files for detr

categories = [
    {
        "supercategory": "with_mask",
        "name": "with_mask",
        "id": 0
    },
    {
        "supercategory": "without_mask",
        "name": "without_mask",
        "id": 1
    }
]

if not os.path.isdir('train_val_split'):
    os.mkdir('train_val_split')

    ### first, split the dataset into train and val
    train = []
    val = []
    images = os.listdir('images')
    for filesname in images:
        n = np.random.uniform(0, 1)
        if n  < 0.7: # training set
            train.append(filesname.split('.')[0])
        else:
            val.append(filesname.split('.')[0])

    with open('train_val_split/train.txt','w') as f:
        f.writelines("%s\n" % a for a in train)

    with open('train_val_split/val.txt','w') as f:
        f.writelines('%s\n' % a for a in val)


### code the facemask dataset into coco-like json file

phases = ["train", "val"]
for phase in phases:

    gt_path = os.path.join("train_val_split/{}.txt".format(phase))
    json_file = "{}.json".format(phase)

    filename = open(gt_path, 'r').read()
    filename = filename.split('\n')[:-1]

    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }
    processed = 0
    annot_count = 0
    ### for each file
    for i, f in enumerate(filename):

        ### load xml file
        tree = ET.parse("annotations/{}.xml".format(f))
        root = tree.getroot()

        ### image information
        img_elem = {"file_name": root.find('filename').text,
                    "height": int(root.find("size/height").text),
                    "width": int(root.find("size/width").text),
                    "id": i}
        res_file["images"].append(img_elem)

        ### for each boundingbox in given image
        for boxes in root.iter('object'):

            ## the 4 coordinate for each bounding box
            ymin, xmin, ymax, xmax = None, None, None, None
            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            w = xmax - xmin
            h = ymax - ymin
            area = w * h
            poly = [[xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]]
            if boxes.find("name").text == 'without_mask':
                cat_id = 1
            else:
                cat_id = 0
            annot_elem = {
                "id": annot_count,
                "bbox": [
                    float(xmin),
                    float(ymin),
                    float(w),
                    float(h)
                ],
                "segmentation": list([poly]),
                "image_id": i,
                "ignore": 0,
                "category_id": cat_id,
                "iscrowd": 0,
                "area": float(area)
            }
            res_file["annotations"].append(annot_elem)
            annot_count += 1
        processed += 1

    ### save into json file
    with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)

    print("Processed {} {} images...".format(processed, phase))
print("Done.")
import os
import xml.etree.ElementTree as ET

# Eğer birden fazla sınıf varsa bu listeye ekle
classes = ["apple"]

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

# Yol ayarları: üst klasöre göre ayarla
annotations_path = '/Users/cagatay/Documents/comp.eng/comp.vision/yolo/annotations/Annotations/Annotations'
output_path = '../labels'
if not os.path.exists(output_path):
    os.makedirs(output_path)

for file in os.listdir(annotations_path):
    if not file.endswith(".xml"):
        continue

    in_file = open(os.path.join(annotations_path, file), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    out_file = open(os.path.join(output_path, file.replace(".xml", ".txt")), 'w')

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)

        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} {' '.join(format(x, '.6f') for x in bb)}\n")

    in_file.close()
    out_file.close()

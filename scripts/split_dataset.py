import os
import random
import shutil

# Kök dizine göre ayar
images_dir = '../images/JPEGImages/JPEGImages'
labels_dir = '../labels'

output_img_train = '../images/train'
output_img_val = '../images/val'
output_lbl_train = '../labels/train'
output_lbl_val = '../labels/val'

# Oran
split_ratio = 0.8

# Çıkış klasörleri oluştur
for path in [output_img_train, output_img_val, output_lbl_train, output_lbl_val]:
    os.makedirs(path, exist_ok=True)

images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
random.shuffle(images)

split_idx = int(len(images) * split_ratio)
train_files = images[:split_idx]
val_files = images[split_idx:]

def move_files(file_list, target_img_dir, target_lbl_dir):
    for file in file_list:
        img_src = os.path.join(images_dir, file)
        lbl_src = os.path.join(labels_dir, file.replace('.jpg', '.txt'))

        shutil.copy2(img_src, os.path.join(target_img_dir, file))
        shutil.copy2(lbl_src, os.path.join(target_lbl_dir, file.replace('.jpg', '.txt')))

move_files(train_files, output_img_train, output_lbl_train)
move_files(val_files, output_img_val, output_lbl_val)

print("✅ Veri başarıyla bölündü.")

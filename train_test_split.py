import os
from shutil import copyfile

path = "./dataset"
train_size = 0.8
allimg_dir = os.path.join(path, "full")
train_dir = os.path.join(path, "train")
test_dir = os.path.join(path, "test")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

length = len(os.listdir(allimg_dir))

train_len = int(length * train_size)

allimg_paths = os.listdir(allimg_dir)
for i in range(0, train_len):
    source_path = os.path.join(allimg_dir, allimg_paths[i])
    target_path = os.path.join(train_dir, allimg_paths[i])
    copyfile(source_path, target_path)

for i in range(train_len, length):
    source_path = os.path.join(allimg_dir, allimg_paths[i])
    target_path = os.path.join(test_dir, allimg_paths[i])
    copyfile(source_path, target_path)
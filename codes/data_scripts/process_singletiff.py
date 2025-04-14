# process folder of tiff files, extract each frame and save as png, and save 1/2 resolution for LQ

import os, tifffile, cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_folder', type=str, help='input folder')
parser.add_argument('output_folder', type=str, help='output folder')
args = parser.parse_args()

from tqdm import tqdm
for file in tqdm(os.listdir(args.input_folder)):
    img = tifffile.imread(os.path.join(args.input_folder, file)).astype(np.float32)/4095*255
    print(img.max(), img.min())

    T, H, W = img.shape

    lq_folder = os.path.join(args.output_folder, f"lq/{file}")
    if not os.path.exists(lq_folder):
        os.makedirs(lq_folder)

    gt_folder = os.path.join(args.output_folder, f"gt/{file}")
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)

    for t in tqdm(range(T)):
        cv2.imwrite(os.path.join(gt_folder, '%08d.png'%t), (img[t]).astype(np.uint8))

        if t % 2 == 1:
            continue

        out = cv2.resize(img[t], (W//2, H//2))
        cv2.imwrite(os.path.join(lq_folder, '%08d.png'%t), (out).astype(np.uint8))
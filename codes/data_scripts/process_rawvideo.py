# process raw video frames into vimeo format with 540P resolution
import os, cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_folder', type=str, help='input folder')
parser.add_argument('output_folder', type=str, help='output folder')
args = parser.parse_args()

lq_folder = os.path.join(args.input_folder, 'indoor_raw_noisy')
gt_folder = os.path.join(args.input_folder, 'indoor_raw_gt')
output_folder = args.output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

from tqdm import tqdm
for scene in tqdm(os.listdir(lq_folder)):
    if not os.path.isdir(os.path.join(lq_folder, scene)):
        continue
    scene_folder = os.path.join(lq_folder, scene)
    if scene < 'scene7':
        continue
    for iso in tqdm(os.listdir(scene_folder)):
        iso_folder = os.path.join(scene_folder, iso)
        for frame in range(1,8):
            filename = f"frame{frame}_noisy9.tiff"
            img = os.path.join(iso_folder, filename)
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_GBRG2BGR)
            img = img.astype(np.float32)
            img = (img-240)/(2**12-1-240)
            for h in range(2):
                for w in range(2):
                    output_folder = os.path.join(args.output_folder, f"lq/{iso}/{scene}_{h}_{w}")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    out = img[h*540:(h+1)*540, w*960:(w+1)*960]

                    #downsample
                    out = cv2.resize(out, (960//2, 540//2), interpolation=cv2.INTER_CUBIC)

                    cv2.imwrite(os.path.join(output_folder, f'im{frame}.png'), (out*255).astype(np.uint8))

for scene in tqdm(os.listdir(gt_folder)):
    if not os.path.isdir(os.path.join(gt_folder, scene)):
        continue
    scene_folder = os.path.join(gt_folder, scene)
    if scene < 'scene7':
        continue
    for iso in tqdm(os.listdir(scene_folder)):
        iso_folder = os.path.join(scene_folder, iso)
        for frame in range(1,8):
            filename = f"frame{frame}_clean_and_slightly_denoised.tiff"
            img = os.path.join(iso_folder, filename)
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img.astype(np.uint16), cv2.COLOR_BAYER_GBRG2BGR)
            img = img.astype(np.float32)
            img = (img-240)/(2**12-1-240)
            for h in range(2):
                for w in range(2):
                    output_folder = os.path.join(args.output_folder, f"gt/{iso}/{scene}_{h}_{w}")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    cv2.imwrite(os.path.join(output_folder, f'im{frame}.png'), (img[h*540:(h+1)*540, w*960:(w+1)*960]*255).astype(np.uint8))

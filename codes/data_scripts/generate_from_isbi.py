# make different SNR noisy image from clean image

# read pure noise images
import os
import tifffile
import argparse
import numpy as np
from typing import List

parser = argparse.ArgumentParser(description='Add noise to each pixel in tiff image')
parser.add_argument('input_folder', type=str, help='input folder, must snr 15 folder')
parser.add_argument('noise_folder', type=str, help='noise folder')
parser.add_argument('--snr', type=List[int], default=[3,4,7,10], help='signal to noise ratio')
args = parser.parse_args()

# SNR = I0 - Ib / sqrt(I0)
Ib = 10

# make clean images
input_folder = args.input_folder
output_folder = args.input_folder + '_clean'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith('.tif'):
        img = tifffile.imread(os.path.join(input_folder, filename)).astype('float32')
        img = img - 50
        img[img < 0] = 0
        img[img > 0] += 50
        tifffile.imwrite(os.path.join(output_folder, filename), img.astype('uint8'))
        print('Saved Clean', os.path.join(output_folder, filename))

# make noisy images
input_folder = output_folder
noise_folder = args.noise_folder
for snr in args.snr:
    output_folder = input_folder + '_noisy_snr' + str(snr)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    imgs_noisy = []
    imgs = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.tif'):
            img = tifffile.imread(os.path.join(input_folder, filename)).astype('float32')

            # read noise image
            filename = filename.replace('snr 15', 'snr 0')
            noise = tifffile.imread(os.path.join(noise_folder, filename)).astype('float32')

            I0 = (snr * snr + 2 * Ib + snr * np.sqrt(snr * snr + 4 * Ib)) / 2

            img_noisy = img / img.max() * I0 + noise
            # tifffile.imwrite(os.path.join(output_folder, filename), img_noisy.astype('uint8'))

            img = img / img.max() * I0
            img[img < 0] = 0
            img[img > 0] += Ib
            # tifffile.imwrite(os.path.join(output_folder, filename.replace('.tif', '_clean.tif')), img.astype('uint8'))

            print(f'Saved Noisy SNR {snr}', os.path.join(output_folder, filename))
            imgs_noisy.append(img_noisy)
            imgs.append(img)

    # concatenate all noisy images into one tif
    os.makedirs(output_folder+"_single", exist_ok=True)
    tifffile.imwrite(os.path.join(output_folder+"_single", f'{os.path.basename(output_folder)}.tif'), imgs_noisy[0].astype('uint8'))
    print(f'Concate Noisy SNR {snr}', os.path.join(output_folder+"_single", f'{os.path.basename(output_folder)}.tif'))

    # concatenate all clean images into one tif
    os.makedirs(output_folder+"_single_clean", exist_ok=True)
    tifffile.imwrite(os.path.join(output_folder+"_single_clean", f'{os.path.basename(output_folder)}.tif'), imgs[0].astype('uint8'))
    print(f'Concate Clean SNR {snr}', os.path.join(output_folder+"_single_clean", f'{os.path.basename(output_folder)}.tif'))


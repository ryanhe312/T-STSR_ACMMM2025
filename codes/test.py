'''
test Zooming Slow-Mo models on arbitrary datasets
write to txt log file
[kosame] TODO: update the test script to the newest version
'''

import os
import os.path as osp
import glob
import logging
import argparse
import numpy as np
import cv2
import torch
import math
from tqdm import tqdm
import utils.util as util
import data.util as data_util

from models.modules import Sakuya_arch, Sakuya_arch_microscopy, Sakuya_arch_microscopy_mamba

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""
    
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    #print('minmax: ', mi, ma)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)
    #print('normalize_mi_ma_debug: ', mi, ma-mi)

    if clip:
        x = np.clip(x, 0, 1)
    
    return x

def read_image(img_path):
    '''read one image from img_path
    Return img: HWC, BGR, [0,1], numpy
    '''
    img_GT = cv2.imread(img_path)
    img = img_GT.astype(np.float32) / 255.
    return img


def read_seq_imgs(img_seq_path, ext='png'):
    '''read a sequence of images'''
    img_path_l = glob.glob(img_seq_path + f'/*.{ext}')
    img_path_l.sort()
    return read_seq_imgs_by_list(img_path_l)

def read_seq_imgs_by_list(img_path_l):
    '''read a sequence of images from the given list'''
    img_l = [read_image(v) for v in img_path_l]
    # stack to TCHW, RGB, [0,1], torch
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(
        np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs

def main(args):
    scale = args.scale
    N_ot = 3 #7
    N_in = 1+ N_ot // 2

    #### model 
    model_path = args.model_path

    if args.model_type == 'Sakuya_arch_microscopy':
        model = Sakuya_arch_microscopy.LunaTokis(64, N_ot, 8, 5, 40)
    elif args.model_type == 'Sakuya_arch_microscopy_mamba':
        model = Sakuya_arch_microscopy_mamba.LunaTokis(64, N_ot, 8, 5, 40)
    elif args.model_type == 'Sakuya_arch_microscopy_mamba2':
        model = Sakuya_arch_microscopy_mamba.LunaTokis(64, N_ot, 8, 5, 40, vssm=2)
    else:
        model = Sakuya_arch.LunaTokis(64, N_ot, 8, 5, 40)

    #### dataset
    data_mode = 'Custom'

    if data_mode == 'Custom':
        test_dataset_folder = args.input_folder # TODO: put your own data path here
    else:
        raise ValueError('Please specify the dataset first.')

    #### evaluation
    flip_test = False #True#
    crop_border = 0

    # temporal padding mode
    padding = 'replicate'
    save_imgs = args.save_imgs

    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
    else:
        device = torch.device('cpu')

    save_folder = args.output_folder
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    model_params = util.get_model_total_params(model)

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))

    def single_forward(model, imgs_in):
        with torch.no_grad():
            # imgs_in.size(): [1,n,3,h,w]
            b,n,c,h,w = imgs_in.size()
            h_n = int(4*np.ceil(h/4))
            w_n = int(4*np.ceil(w/4))
            imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
            imgs_temp[:,:,:,0:h,0:w] = imgs_in
            model_output = model(imgs_temp)
            model_output = model_output[:, :, :, 0:scale*h, 0:scale*w]
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    sub_folder_l = sorted(glob.glob(test_dataset_folder))
    # print(sub_folder_l)

    if model_path: model.load_state_dict(torch.load(model_path))

    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_psnr_y_l = []
    avg_ssim_l = []
    avg_ssim_y_l = []
    sub_folder_name_l = []
    # total_time = []
    # for each sub-folder
    for sub_folder in tqdm(sub_folder_l):
        gt_tested_list = []
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)
        img_LR_l = sorted(glob.glob(sub_folder + '/*'))

        if save_imgs:
            util.mkdirs(save_sub_folder)

            #### read LR images
            imgs = read_seq_imgs(sub_folder, args.ext)
            # print(f"Sequence length {len(imgs)}, width {imgs.shape[3]}, height {imgs.shape[2]}")

        #### read GT images
        img_GT_l = []
        sub_folder_GT = osp.join(args.reference_folder, sub_folder_name)

        if osp.exists(sub_folder_GT):
            for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT,'*.'+args.ext))):
                # print(img_GT_path)
                img_GT_l.append(util.read_image(img_GT_path))

        avg_psnr, avg_psnr_sum, cal_n = 0,0,0
        avg_psnr_y, avg_psnr_sum_y = 0,0
        avg_ssim, avg_ssim_sum = 0,0
        avg_ssim_y, avg_ssim_sum_y = 0,0
        
        if len(img_LR_l) == len(img_GT_l):
            skip = True
        else:
            skip = False
        
        select_idx_list = util.test_index_generation(skip, N_ot, len(img_LR_l))
        
        # process each image
        for select_idxs in select_idx_list:
            # get input images
            select_idx = select_idxs[0]
            gt_idx = select_idxs[1]
            
            if save_imgs: 
                imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
                output = single_forward(model, imgs_in)
                outputs = output.data.float().cpu().squeeze(0)            

            if flip_test:
                # flip W
                output = single_forward(model, torch.flip(imgs_in, (-1, )))
                output = torch.flip(output, (-1, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip H
                output = single_forward(model, torch.flip(imgs_in, (-2, )))
                output = torch.flip(output, (-2, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip both H and W
                output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                output = torch.flip(output, (-2, -1))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output

                outputs = outputs / 4

            # save imgs
            for idx, name_idx in enumerate(gt_idx):
                if name_idx in gt_tested_list:
                    continue
                gt_tested_list.append(name_idx)

                if save_imgs: 
                  output_f = outputs[idx,:,:,:].squeeze(0)
                  output = util.tensor2img(output_f)
                  
                  corr = args.gamma
                  if corr:    # perform gamma correction because interpolated images have different brightness
                      if (idx % 2) == 0:
                        brightness_even = output.mean()
                      else:
                        output = cv2.blur(output,(3,3))
                        brightness_odd = output.mean()
                        gamma = math.sqrt(brightness_even/brightness_odd)
                        # gamma =brightness_even/brightness_odd
                        output = np.power(output, gamma)
                        # output = output*gamma
                        corrected_brightness = output.mean()
                        print(f"gamma {gamma} brightness_odd {brightness_odd} brightness_even {brightness_even} corrected_brightness {corrected_brightness} ")

                  cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(name_idx+1)), output)

                if osp.exists(sub_folder_GT):
                    #### calculate PSNR
                    output = output / 255.

                    GT = np.copy(img_GT_l[name_idx])

                    if crop_border == 0:
                        cropped_output = output
                        cropped_GT = GT
                    else:
                        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
                        cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border, :]

                    # normalize
                    cropped_output = normalize(cropped_output, 0, 100, clip=True)
                    cropped_GT = normalize(cropped_GT, 0, 100, clip=True)

                    cropped_output = cv2.resize(cropped_output, (cropped_GT.shape[1], cropped_GT.shape[0]), interpolation=cv2.INTER_CUBIC)
                    cropped_GT_y = data_util.bgr2ycbcr(cropped_GT, only_y=True)
                    cropped_output_y = data_util.bgr2ycbcr(cropped_output, only_y=True)
                    
                    crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
                    crt_psnr_y = util.calculate_psnr(cropped_output_y * 255, cropped_GT_y * 255)
                    crt_ssim = util.calculate_ssim(cropped_output * 255, cropped_GT * 255)
                    crt_ssim_y = util.calculate_ssim(cropped_output_y * 255, cropped_GT_y * 255)
                    # logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB  PSNR-Y: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_psnr, crt_psnr_y))
                    # logger.info('{:3d} - {:25}.png \tSSIM: {:.6f} dB  SSIM-Y: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_ssim, crt_ssim_y))
                    avg_psnr_sum += crt_psnr
                    avg_psnr_sum_y += crt_psnr_y
                    avg_ssim_sum += crt_ssim
                    avg_ssim_sum_y += crt_ssim_y
                    cal_n += 1

        if cal_n != 0:
            avg_psnr = avg_psnr_sum / cal_n
            avg_psnr_y = avg_psnr_sum_y / cal_n
            avg_ssim = avg_ssim_sum / cal_n
            avg_ssim_y = avg_ssim_sum_y / cal_n
            # logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} frames; '.format(sub_folder_name, avg_psnr, avg_psnr_y, cal_n))
            # logger.info('Folder {} - Average SSIM: {:.6f} dB SSIM-Y: {:.6f} dB for {} frames; '.format(sub_folder_name, avg_ssim, avg_ssim_y, cal_n))
            avg_psnr_l.append(avg_psnr)
            avg_ssim_l.append(avg_ssim)
            avg_psnr_y_l.append(avg_psnr_y)
            avg_ssim_y_l.append(avg_ssim_y)

    if len(avg_psnr_l) > 0:
        logger.info('################ Final Results ################')
        logger.info('Total PSNR: {:.6f} dB. '.format(sum(avg_psnr_y_l) / len(avg_psnr_y_l)))
        logger.info('Total SSIM: {:.6f} dB. '.format(sum(avg_ssim_y_l) / len(avg_ssim_y_l)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--scale', type=int, default=2, help='Select scale factor of the model')
    parser.add_argument('-v','--save_imgs', action='store_true', help='Save output images')
    parser.add_argument('-g','--gamma', action='store_true', help='Perform gamma correction')
    parser.add_argument('-c','--channel', type=int, default=1, help='Select zoom factor of the image')
    parser.add_argument('-m','--model_path', type=str, help='Select input model path')
    parser.add_argument('-t','--model_type', type=str, help='Select input model type')
    parser.add_argument('-i','--input_folder', type=str, help='Path to the input folder')
    parser.add_argument('-o','--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('-r','--reference_folder', type=str, default=None, help='Path to the gt folder')
    parser.add_argument('-e','--ext', type=str, default='png', help='Path to the gt folder')
    args = parser.parse_args()
    main(args)

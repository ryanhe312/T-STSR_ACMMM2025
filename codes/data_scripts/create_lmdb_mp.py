'''create lmdb files for Vimeo90K-7 frames training dataset (multiprocessing)
Will read all the images to the memory
'''

import os
import sys
import os.path as osp
import glob
import pickle
import lmdb
import cv2
import shutil
from tqdm import tqdm

def reading_image_worker(path, key):
    '''worker for reading images'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def save_to_lmbd(img_folder, test_or_train, H_dst, W_dst, batch, mode, scale_factor):
    '''create lmdb for the Vimeo90K-7 frames dataset, each image with fixed size
    GT: [3, 256, 448]
        Only need the 4th frame currently, e.g., 00001_0001_4
    LR: [3, 64, 112]
        With 1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001
    '''
    #### configurations
    n_thread = 40

    # define the septest/trainlist & lmdb_save_path
    # path_parent = os.path.dirname(img_folder)

    if test_or_train == "test":
      txt_file = os.path.join(img_folder,"sep_testlist.txt")
      lmdb_save_path = os.path.join(img_folder, f"vimeo7_{test_or_train}_x{scale_factor}_{mode}.lmdb")
      img_folder_selected = os.path.join(img_folder, f"test_{scale_factor}")
      if os.path.isdir(lmdb_save_path):
        shutil.rmtree(lmdb_save_path)
    if test_or_train == "train":
      txt_file = os.path.join(img_folder,"sep_trainlist.txt")
      lmdb_save_path = os.path.join(img_folder, f"vimeo7_{test_or_train}_x{scale_factor}_{mode}.lmdb")
      img_folder_selected = os.path.join(img_folder, f"train_{scale_factor}")
      if os.path.isdir(lmdb_save_path):
        shutil.rmtree(lmdb_save_path)

    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in tqdm(train_l):
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        file_l = glob.glob(osp.join(img_folder_selected, folder, sub_folder) + '/*')
        all_img_list.extend(file_l)
        for j in range(7):
            keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'HR': 
        all_img_list = [v for v in all_img_list if v.endswith('.png')]
        keys = [v for v in keys]

    print('Calculating the total size of images...')
    data_size = sum(os.stat(v).st_size for v in all_img_list)

    #### read all images to memory (multiprocessing)
    print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
    
    #### create lmdb environment
    env = lmdb.open(lmdb_save_path, map_size=data_size * 30)
    txn = env.begin(write=True)  # txn is a Transaction object

    #### write data to lmdb
    #pbar = util.ProgressBar(len(all_img_list))

    i = 0
    for path, key in tqdm(zip(all_img_list, keys)):
        #pbar.update('Write {}'.format(key))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)        
        key_byte = key.encode('ascii')
        H, W, C = img.shape  # fixed shape
        assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, img)
        i += 1
        if  i % batch == 1:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish reading and writing {} images.'.format(len(all_img_list)))
            
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    if mode == 'HR':
        meta_info['name'] = 'Vimeo7_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'Vimeo7_train_LR7'
    meta_info['resolution'] = '{}_{}_{}'.format(3, H_dst, W_dst)
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = key_set
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'Vimeo7_train_keys.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


if __name__ == "__main__":
    save_HR = "/home/user2/dataset/interpolation/vimeo_septuplet/Out_HR_LR/HR"
    save_LR = "/home/user2/dataset/interpolation/vimeo_septuplet/Out_HR_LR/LR"

    HR_h = 256
    HR_w = 448
    LR_h = 128
    LR_w = 224

    # if error occurs the batch value can be changed
    batch = 3000
    scale_factor = 2


    test_or_train = "train"
    mode = "HR"
    save_to_lmbd(save_HR, test_or_train, HR_h, HR_w, batch, mode, scale_factor)

    mode = "LR"
    save_to_lmbd(save_LR, test_or_train, LR_h, LR_w, batch, mode, scale_factor)
    
    test_or_train = "test"
    mode = "HR"
    save_to_lmbd(save_HR, test_or_train, HR_h, HR_w, batch, mode, scale_factor)
    
    mode = "LR"
    save_to_lmbd(save_LR, test_or_train, LR_h, LR_w, batch, mode, scale_factor)

    train_LMBD_HR = save_HR + "/vimeo7_train_x{}_HR.lmdb".format(scale_factor)
    train_LMBD_LR = save_LR + "/vimeo7_train_x{}_LR.lmdb".format(scale_factor)

    test_LMBD_HR = save_HR + "/vimeo7_test_x{}_HR.lmdb".format(scale_factor)
    test_LMBD_LR = save_LR + "/vimeo7_test_x{}_LR.lmdb".format(scale_factor)
    print(train_LMBD_HR, train_LMBD_LR, test_LMBD_HR, test_LMBD_LR) 
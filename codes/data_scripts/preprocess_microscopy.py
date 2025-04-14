import os, sys
import os.path as osp
import cv2
import numpy as np
import shutil
import math
import random
import argparse
import glob
import pickle
import lmdb
from tqdm import tqdm
from skimage import io
try: 
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass

random.seed(1234)

def get_img_dim(img):
  """This function gets the right x,y,t,z dimensions and if it is an RGB image or not"""
  if (img.shape[-1] ==3 and len(img.shape) ==5):
    use_RGB = True
    t_dim, z_dim, y_dim, x_dim, _ = img.shape
  elif (img.shape[-1] ==3 and len(img.shape) ==4):
    use_RGB = True
    t_dim, y_dim, x_dim, channel = img.shape
    zeros = np.zeros((t_dim,1,y_dim,x_dim,channel))
    zeros[:,0,:,:,:] = img
    img = zeros
    t_dim, z_dim, y_dim, x_dim, _ = img.shape
  elif (img.shape[-1] !=3 and len(img.shape) ==3):  # create a 4th dimension
    use_RGB = False
    t, y, x = img.shape
    zeros = np.zeros((t,1,y,x))
    zeros[:,0,:,:] = img
    img = zeros
    t_dim, z_dim, y_dim, x_dim= img.shape
  elif (img.shape[-1] !=3 and len(img.shape) ==4):
    use_RGB = False
    t_dim, z_dim, y_dim, x_dim = img.shape
  return t_dim, z_dim, y_dim, x_dim, use_RGB

def split_test_train_sequences_data(inPath, outPath, guide):
  """This function splits the sequences folder into the test and train folder with the given format
  based on the guide txt files"""
  #if os.path.isdir(outPath):
  #  shutil.rmtree(outPath)
  f = open(guide, "r")
  lines = f.readlines()
  for l in tqdm(lines):
      line = l.replace('\n','')
      this_folder = os.path.join(inPath, line)
      dest_folder = os.path.join(outPath, line)
      if os.path.exists(dest_folder):
        print(f"Folder already moved: {dest_folder}")
      else:
        shutil.move(this_folder, dest_folder)
  print('Done')

def prep_folder_structure(new_path, guide_path):
  '''this function creates the same folder and subfolder structure as provided in the sequences folder in a 
  new given new_location path based on a master_sep_guide.txt file which recombined all folders from test and train'''
  print(f"Prepare Folder structure: {new_path}")
  with open(osp.join(guide_path,"master_sep_guide.txt"), "r") as temp:
    for line in tqdm(temp):
        one = line[:-1].split("/")[0]
        two = line[:-1].split("/")[1]
        folder_1 = os.path.join(new_path, one)
        if not os.path.exists(folder_1):
          os.mkdir(folder_1)
          folder_2 = os.path.join(folder_1, two)
          os.mkdir(folder_2)
        else:
          folder_2 = os.path.join(folder_1, two)
          os.mkdir(folder_2)

def get_all_filepaths(input_path, N_frames, guide_path):
    '''This function gets the paths based on the folder and the N_frames provided'''
    print("Execute: get_all_filepaths")
    flist = []
    with open(osp.join(guide_path,"master_sep_guide.txt"), "r") as temp:
      for line in tqdm(temp):
        folder_path = os.path.join(input_path,line[:-1])
        for i in range(1,N_frames+1):
          file_name = f"im{i}.png"
          file_path = os.path.join(folder_path, file_name)
          flist.append(file_path)
    return flist

def get_all_filepaths_in_folder(folder_path):
    '''This function gets the paths from each file in folder and subfolder of a given location'''
    # flist = []
    # for path, subdirs, files in tqdm(os.walk(folder_path)):
    #       for name in files:
    #         flist.append(os.path.join(path, name))
    return [os.path.join(folder_path, name) for name in os.listdir(folder_path) if '.tif' in name]

def create_folder_list_from_txt_guide(testlist_txt, trainlist_txt, guide_path):
    print("Execute: create_folder_list_from_txt_guide")
    print(f"testlist_txt: {testlist_txt}")
    print(f"trainlist_txt: {trainlist_txt}")
    list_path_list = []
    with open(testlist_txt, "r") as f:
      for line in f:
        list_path_list.append(line)
    with open(trainlist_txt, "r") as f:
      for line in f:
        list_path_list.append(line)
    list_path_list.sort()

    print("Number of files: ", len(list_path_list))
    print(f"Save master_sep_guide.txt to {osp.join(guide_path,'master_sep_guide.txt')}")
    with open(osp.join(guide_path,"master_sep_guide.txt"), "w") as temp:
      for line in list_path_list:
        temp.write(line)


def generate_mod_LR(up_scale, sourcedir, savedir, train_guide, test_guide, continue_loading, N_frames, log_path):
    """This function generates the high and low resulution images in a given output folder"""

    create_folder_list_from_txt_guide(train_guide, test_guide, savedir)

    save_HR = os.path.join(savedir, 'HR')
    save_LR = os.path.join(savedir, 'LR')
 
    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(up_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
      
    # Create folder system
    if continue_loading == False:
        print("Restart loading")
        os.makedirs(savedir, exist_ok=True)

        os.makedirs(save_HR, exist_ok=True)
        os.makedirs(save_LR, exist_ok=True)
        
        os.makedirs(saveHRpath, exist_ok=True)
        prep_folder_structure(saveHRpath, savedir)

        os.makedirs(saveLRpath, exist_ok=True)
        prep_folder_structure(saveLRpath, savedir)

        # copy the set_guide text files in each folder (HR, LR)
        train_guide_HR = saveHRpath[:-3]+"/sep_trainlist.txt"
        train_guide_LR = saveLRpath[:-3]+"/sep_trainlist.txt"

        test_guide_HR = saveHRpath[:-3]+"/sep_testlist.txt"
        test_guide_LR = saveLRpath[:-3]+"/sep_testlist.txt"

        shutil.copy(train_guide, train_guide_HR)
        shutil.copy(train_guide, train_guide_LR)

        shutil.copy(test_guide, test_guide_HR)
        shutil.copy(test_guide, test_guide_LR)
        with open(log_path, "w") as f:
            f.write("start")
        with open(log_path, "a") as f:
            f.write(f'Created new folders: {savedir} \n')
            f.write(f'Created new folders: {save_HR}\n')
            f.write(f'Created new folders: {save_LR}\n')
            f.write(f'Created new folders: {saveHRpath}\n')
            f.write(f'Created new file: {train_guide_HR}\n')
            f.write(f'Created new file: {test_guide_LR}\n')
    else:
        with open(log_path, "w") as f:
            f.write("start")
    filepaths = get_all_filepaths(sourcedir, N_frames, savedir)
    print(f"number of files: {len(filepaths)}")
    num_files = len(filepaths)

 # # prepare data with augementation
    for i in tqdm(range(num_files)):
        filename = filepaths[i]
        file_folder_path = filename[-18:]
        # check if file was already processed
        file_checker_path = os.path.join(saveHRpath, file_folder_path)
        if os.path.exists(file_checker_path):
          with open(log_path, "a") as f:
            f.write(f"File already exists: {file_checker_path}\n")
          continue
        else: 
          try:
            with open(log_path, "a") as f:
              f.write('No.{} -- Processing {}\n'.format(i, filename))
            # read image
            image = cv2.imread(filename)

            width = int(np.floor(image.shape[1] / up_scale))
            height = int(np.floor(image.shape[0] / up_scale))
            # modcrop
            if len(image.shape) == 3:
                image_HR = image[0:up_scale * height, 0:up_scale * width, :]
            else:
                image_HR = image[0:up_scale * height, 0:up_scale * width]
            # LR
            image_LR = imresize_np(image_HR, 1 / up_scale, True)
            file_folder_path = filename[-18:]
            cv2.imwrite(os.path.join(saveHRpath, file_folder_path), image_HR)
            cv2.imwrite(os.path.join(saveLRpath, file_folder_path), image_LR)
          except:
            with open(log_path, "a") as f:
              f.write('No.{} -- failed {}\n'.format(i, filename))     

    return save_HR, save_LR


#############################Prepare LMBD data ##################################
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
    env = lmdb.open(lmdb_save_path, map_size=data_size * 200)
    txn = env.begin(write=True)  # txn is a Transaction object

    #### write data to lmdb
    #pbar = util.ProgressBar(len(all_img_list))

    i = 0
    for path, key in tqdm(list(zip(all_img_list, keys))):
        #pbar.update('Write {}'.format(key))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)        
        key_byte = key.encode('ascii')
        H, W, C = img.shape  # fixed shape
        assert H == H_dst and W == W_dst and C == 3, f'different shape.{H, H_dst, W, W_dst, C}'
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
  
import shutil
import os
from tqdm import tqdm
  
def run_split_sequence(mode, scale_factor, save_HR, save_LR, test_or_train, outPath_test):
    if mode == "HR":
      sequences_path = os.path.join(save_HR, f"x{scale_factor}")
      train_guide = os.path.join(save_HR, "sep_trainlist.txt")
      test_guide = os.path.join(save_HR, "sep_testlist.txt")
      if test_or_train == "test":
          outPath_test = os.path.join(save_HR, f"test_{scale_factor}")
          split_test_train_sequences_data(sequences_path, outPath_test, test_guide)
      if test_or_train == "train":
          outPath_train = os.path.join(save_HR, f"train_{scale_factor}")
          split_test_train_sequences_data(sequences_path, outPath_train, train_guide)

    if mode == "LR":
      sequences_path = os.path.join(save_LR, f"x{scale_factor}")
      train_guide = os.path.join(save_LR, "sep_trainlist.txt")
      test_guide = os.path.join(save_LR, "sep_testlist.txt")
      if test_or_train == "test":
          outPath_test = os.path.join(save_LR, f"test_{scale_factor}")
          split_test_train_sequences_data(sequences_path, outPath_test, test_guide)
      if test_or_train == "train":
          outPath_train = os.path.join(save_LR, f"train_{scale_factor}")
          split_test_train_sequences_data(sequences_path, outPath_train, train_guide)
    return sequences_path, train_guide, test_guide, outPath_test

  
  
def prepare_lmbd(save_HR, save_LR, HR_input_h, HR_input_w, scale_factor, batch):
  
  LR_input_h = HR_input_h/scale_factor
  LR_input_w = HR_input_w/scale_factor

  test_or_train = "train"
  mode = "HR"
  save_to_lmbd(save_HR, test_or_train, HR_input_h, HR_input_w, batch, mode, scale_factor)

  mode = "LR"
  save_to_lmbd(save_LR, test_or_train, LR_input_h, LR_input_w, batch, mode, scale_factor)
  
  test_or_train = "test"
  mode = "HR"
  save_to_lmbd(save_HR, test_or_train, HR_input_h, HR_input_w, batch, mode, scale_factor)
  
  mode = "LR"
  save_to_lmbd(save_LR, test_or_train, LR_input_h, LR_input_w, batch, mode, scale_factor)

  train_LMBD_HR = save_HR + "/vimeo7_train_x{}_HR.lmdb".format(scale_factor)
  train_LMBD_LR = save_LR + "/vimeo7_train_x{}_LR.lmdb".format(scale_factor)

  test_LMBD_HR = save_HR + "/vimeo7_test_x{}_HR.lmdb".format(scale_factor)
  test_LMBD_LR = save_LR + "/vimeo7_test_x{}_LR.lmdb".format(scale_factor)
  return train_LMBD_HR, train_LMBD_LR, test_LMBD_HR, test_LMBD_LR, LR_input_h


def correct_channels(img):
  '''For 2D + T (with or without RGB) a artificial z channel gets created'''
  if img.shape[-1] ==3:
    use_RGB = True
  else:
    use_RGB = False
  if len(img.shape) ==4 and use_RGB:
    t, x, y, c = img.shape
    zeros = np.zeros((t,1,y,x,c), dtype=np.uint8)
    zeros[:,0,:,:,:] = img
    img = zeros
  elif len(img.shape) ==3 and not use_RGB:
    t, y, x = img.shape
    zeros = np.zeros((t,1,y,x), dtype=np.uint8)
    zeros[:,0,:,:] = img
    img = zeros
  return img, use_RGB

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Path to dataset dir.')
    parser.add_argument('output', type=str, help='Path to dataset dir.')
    args = parser.parse_args()

    ############ FIRST STEP ##################
    ##### Separating the single images #######
    ##########################################

    #@markdown Provide the folder with the training data
    # Define the necessary paths needed later
    Source_path = args.source#@param {type:"string"}
    Parent_path = args.output#@param {type:"string"}
    test_train_seq_path = os.path.join(Parent_path, "sequences")
    os.makedirs(test_train_seq_path, exist_ok=True)

    # Paramenters
    test_train_split = 0
    N_frames = 7

    # create seq_lists *.txt
    train_seq_txt = os.path.join(Parent_path, "sep_trainlist.txt")
    test_seq_txt = os.path.join(Parent_path, "sep_testlist.txt")
    with open(train_seq_txt, "w") as f:
      f.write("")
    with open(test_seq_txt, "w") as f:
      f.write("")

    # delete test_train_folder if already exists
    if os.path.isdir(test_train_seq_path):
      shutil.rmtree(test_train_seq_path)
    os.mkdir(test_train_seq_path)

    #get all files in the selected folder
    flist = get_all_filepaths_in_folder(Source_path)

    # split the different images and save them in the sequence folder with a given folderstructure
    # and create the test train split seq txt files
    for counter_1, file_path in tqdm(enumerate(flist)):
      file_folder = "%05d"%(counter_1+1)
      print(file_folder)
      os.mkdir(os.path.join(test_train_seq_path, file_folder))
      file_folder_path = os.path.join(test_train_seq_path, file_folder)

      img = io.imread(file_path)
      # makes 3D into 4D dataset if needed
      img, _ = correct_channels(img)
      t_dim, z_dim, y_dim, x_dim, use_RGB = get_img_dim(img)

      #calculate how many folders need to be created to cover all the images
      N_folders_per_slice = math.ceil(t_dim/N_frames)
      counter_2 = 1
      for z in tqdm(range(z_dim)):
        if (use_RGB and len(img.shape) ==5) or (use_RGB and len(img.shape) ==4):
          img_slice = img[:, z, :, :, :]
        else:
          img_slice = img[:, z, :, :]

        for seq in tqdm(range(1,N_folders_per_slice)):
          seq_folder = "%04d"%(counter_2)
          seq_folder_path = os.path.join(file_folder_path, seq_folder)
          os.mkdir(seq_folder_path)
          counter_2 += 1
          #create lever to randomly shift samples to test or train depending on the chosen split
          test_train_lever = random.uniform(0, 1)
          if test_train_lever < test_train_split:
            with open(test_seq_txt, "a") as f:
              f.write(f"{file_folder}/{seq_folder}\n")
          else:
            with open(train_seq_txt, "a") as f:
              f.write(f"{file_folder}/{seq_folder}\n")

          #save a given number of images as png in the new folder
          for im_num in range(1, N_frames+1):
            # print(((seq-1)*N_frames+(im_num-1)), y_dim, x_dim)
            png_img_path = os.path.join(seq_folder_path, f"im{im_num}.png")
            if use_RGB:
                png_img = img_slice[((seq-1)*N_frames+(im_num-1)), :, :, :]
                img_channels = png_img
            else:
                png_img = img_slice[((seq-1)*N_frames+(im_num-1)), :, :]
                img_channels = np.zeros((y_dim, x_dim, 3))
                img_channels[:,:,0] = png_img
                img_channels[:,:,1] = png_img
                img_channels[:,:,2] = png_img
            io.imsave(png_img_path, img_channels.astype(np.uint8), check_contrast=False)


    ############ SECOND STEP ##################
    ######## Generate HR LR images ############
    ###########################################
    # This cell creates the following folder sequences in a provided output-path:
    # downscaled low-resolution (LR)
    # The original high-resolution (HR)

    # inPath = "/".join(folder_path.split("/")[:-1])
    ##'ZS4Mic/demo'#@param {type:"string"}
    sequences_path = os.path.join(Parent_path, "sequences")
    # test_or_train = "test"#@param ["test", "train"]

    outPath = os.path.join(Parent_path, "Out_HR_LR")
    scale_factor = 2 #@param ["1", "2", "4"] {type:"raw"}


    train_guide = os.path.join(Parent_path, "sep_trainlist.txt")
    test_guide = os.path.join(Parent_path, "sep_testlist.txt")

    N_frames = 7
    continue_loading = False
    log_path = os.path.join(outPath, "HR_LR_log.txt")
    os.makedirs(outPath, exist_ok=True)
    save_HR, save_LR = generate_mod_LR(scale_factor, sequences_path, outPath, train_guide, test_guide,continue_loading, N_frames, log_path)


    ############ THIRD STEP ###################
    #### Separate Test and Train samples ######
    ###########################################
    # Split sequences folder into test and train data
    # The test data is not used in this notebook but can be used for quick training

    mode = "HR"
    test_or_train = "test"
    outPath_test = os.path.join(save_HR, f"test_{scale_factor}")
    sequences_path, train_guide, test_guide, outPath_test = run_split_sequence(mode, scale_factor, save_HR, save_LR, test_or_train, outPath_test)

    mode = "HR"
    test_or_train = "train"
    outPath_train = os.path.join(save_HR, f"train_{scale_factor}")
    sequences_path, train_guide, test_guide, outPath_test = run_split_sequence(mode, scale_factor, save_HR, save_LR, test_or_train, outPath_test)

    test_or_train = "test"
    mode = "LR"
    outPath_test = os.path.join(save_LR, f"test_{scale_factor}")
    sequences_path, train_guide, test_guide, outPath_test = run_split_sequence(mode, scale_factor, save_HR, save_LR, test_or_train, outPath_test)

    test_or_train = "train"
    mode = "LR"
    outPath_train = os.path.join(save_LR, f"train_{scale_factor}")
    sequences_path, train_guide, test_guide, outPath_test = run_split_sequence(mode, scale_factor, save_HR, save_LR, test_or_train, outPath_test)



    ############ FOURTH STEP ###################
    #### Create LMDB Files for training ########
    ############################################
    # Prepare lmdb folders for faster training
    # This step has to be performed with the HR **! and !** LR images to create the test data and train data folder
    #Select which data should be prepared into LMBD format
    '''create lmdb files for Vimeo90K-7 frames training dataset (multiprocessing)
    Will read all the images to the memory
    '''

    # if error occurs the batch value can be changed
    batch = 3000

    train_LMBD_HR, train_LMBD_LR, test_LMBD_HR, test_LMBD_LR, LR_input_dim  = prepare_lmbd(save_HR, save_LR, y_dim, x_dim, scale_factor, batch)

    shutil.rmtree(test_train_seq_path)
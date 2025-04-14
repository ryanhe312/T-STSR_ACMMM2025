import os
import sys
import cv2
import numpy as np
import os.path as osp
import shutil
from tqdm import tqdm

try:
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass


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
    flist = []
    for path, subdirs, files in tqdm(os.walk(folder_path)):
          for name in files:
            flist.append(os.path.join(path, name))
    return flist

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

            if up_scale == 1:
                file_folder_path = filename[-18:]

                shutil.copy(filename, os.path.join(saveHRpath, file_folder_path))
                shutil.copy(filename, os.path.join(saveLRpath, file_folder_path))
                continue

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

if __name__ == "__main__":
    ############ SECOND STEP ##################
    ######## Generate HR LR images ############
    ###########################################
    # This cell creates the following folder sequences in a provided output-path:
    # downscaled low-resolution (LR)
    # The original high-resolution (HR)
    Parent_path = '/home/user2/dataset/interpolation/vimeo_septuplet'#@param {type:"string"}

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



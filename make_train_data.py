'''
This script creates fMRI and images data for training of end-to-end deep reconstruction.
'''
import csv
import fnmatch
import glob
import os
from datetime import datetime
import pickle
import PIL.Image
import numpy as np
import bdpy
import os
# Settings ---------------------------------------------------------------
# Image size
img_size = (256, 256)
# For image jittering, we prepare the images to be larger than 227 x 227
# fMRI data
fmri_data_table = [
    {'subject': 'sub-01',
     'data_file': './data/mri/sub-01_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1',
     'output_dir': './lmdb/sub-01'},
    {'subject': 'sub-02',
     'data_file': './data/mri/sub-02_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1',
     'output_dir': './lmdb/sub-02'},
    {'subject': 'sub-03',
     'data_file': './data/mri/sub-03_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1',
     'output_dir': './lmdb/sub-03'}
]
# Image data
image_dir = './data/images/training'
image_file_pattern = '*.JPEG'
# Create LMDB data -------------------------------------------------------
for sbj in fmri_data_table:    # Create LMDB for fMRI data
    print('----------------------------------------')
    print('Subject: %s' % sbj['subject'])
    print('')
    if os.path.exists(sbj['output_dir']):
        print('%s overwriting.' % sbj['output_dir'])
    else:
        os.makedirs(sbj['output_dir'])
    # Load fMRI data
    print('Loading %s' % sbj['data_file'])
    fmri_data_bd = bdpy.BData(sbj['data_file'])

    # Load image files
    images_list = glob.glob(os.path.join(image_dir, image_file_pattern))  # List of image files (full path)
    images_table = {os.path.splitext(os.path.basename(f))[0]: f
                    for f in images_list}                                 # Image label to file path table
    label_table = {os.path.splitext(os.path.basename(f))[0]: i + 1
                   for i, f in enumerate(images_list)}                    # Image label to serial number table

    # Get image labels in the fMRI data
    fmri_labels = fmri_data_bd.get('Label')[:, 1].flatten()

    # Convert image labels in fMRI data from float to file name labes (str)
    fmri_labels = ['n%08d_%d' % (int(('%f' % a).split('.')[0]),
                                 int(('%f' % a).split('.')[1]))
                   for a in fmri_labels]

    # Get sample indexes
    n_sample = fmri_data_bd.dataset.shape[0]

    index_start = 1
    index_end = n_sample
    index_step = 1

    sample_index_list = range(index_start, index_end + 1, index_step)

    # Shuffle the sample indexes
    sample_index_list = np.random.permutation(sample_index_list)

    # Save the sample indexes
    save_name = 'sample_index_list.txt'
    np.savetxt(os.path.join(sbj['output_dir'], save_name), sample_index_list, fmt='%d')

    # Get fMRI data in the ROI
    fmri_data = fmri_data_bd.select(sbj['roi_selector'])

    # Normalize fMRI data
    fmri_data_mean = np.mean(fmri_data, axis=0)
    fmri_data_std = np.std(fmri_data, axis=0)

    fmri_data = (fmri_data - fmri_data_mean) / fmri_data_std

    map_size = 100 * 1024 * len(sample_index_list) * 10


    fmri_output_dir = sbj['output_dir']
    myfmri = []    

    for j0, sample_index in np.ndenumerate(sample_index_list):

        sample_label = fmri_labels[sample_index - 1]  # Sample label (file name)
        sample_label_num = label_table[sample_label]  # Sample label (serial number)


        # fMRI data in the sample
        sample_data = fmri_data[sample_index - 1, :]
        sample_data = np.float64(sample_data)  # Datum should be double float (?)
        sample_data = np.reshape(sample_data, (sample_data.size, 1, 1))  # Num voxel x 1 x 1

        datum = [sample_data, sample_label_num]
        myfmri.append(datum)
        
    with open(fmri_output_dir + "/fmri.pkl", 'wb') as f:
        pickle.dump(myfmri, f)


    # Create lmdb for images
    print('----------------------------------------')
    print('Images')

    map_size = 30 * 1024 * len(sample_index_list) * 10

    images_output_dir = sbj['output_dir']
    myimages = []

    # with env.begin(write=True) as txn:
    for j0, sample_index in np.ndenumerate(sample_index_list):

        sample_label = fmri_labels[sample_index - 1]  # Sample label (file name)
        sample_label_num = label_table[sample_label]  # Sample label (serial number)

        # Load images
        image_file = images_table[sample_label]
        img = PIL.Image.open(image_file)
        img = img.resize(img_size, PIL.Image.BICUBIC)
        img = np.array(img)

        # Monochrome --> RGB
        if img.ndim == 2:
            img_rgb = np.zeros((img_size[0], img_size[1], 3), dtype=img.dtype)
            img_rgb[:, :, 0] = img
            img_rgb[:, :, 1] = img
            img_rgb[:, :, 2] = img
            img = img_rgb

        datum = [img, sample_label_num]
        myimages.append(datum)

    with open(fmri_output_dir + "/images.pkl", 'wb') as f:
        pickle.dump(myimages, f)


# Saving our data as np arrays
for i in range(1, 4):
    with open(f'lmdb/sub-0{i}/fmri.pkl', 'rb') as f:
        fmri_raw = pickle.load(f)

    with open(f'lmdb/sub-0{i}/images.pkl', 'rb') as f:
        images = pickle.load(f)

    fmri = [x for x, y in fmri_raw]
    labels = [y for x, y in fmri_raw]
    images = [x for x, y, in images]

    fmri_concat = np.stack( fmri, axis = 0).squeeze()
    labels_concat = np.stack( labels, axis = 0).squeeze()
    images_concat = np.stack( images, axis = 0).squeeze()

    np.save(f'./np_data/sub_0{i}_images_train.npy', images_concat, allow_pickle=True, fix_imports=True)
    np.save(f'./np_data/sub_0{i}_fmri_train.npy', fmri_concat, allow_pickle=True, fix_imports=True)
    np.save(f'./np_data/sub_0{i}_labels_train.npy', labels_concat, allow_pickle=True, fix_imports=True)

# clean the data out
os.system("rm -r lmdb")
print('Done!')
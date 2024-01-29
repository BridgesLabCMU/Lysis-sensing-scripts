import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
import os
import tifffile as tif
import numpy as np
from scipy.ndimage import shift, uniform_filter, binary_fill_holes, distance_transform_edt
import cv2
import shutil
from sklearn.mixture import BayesianGaussianMixture
from skimage.registration import phase_cross_correlation
from skimage.morphology import reconstruction
from skimage.measure import label

# set directory here
directory = 'B:/Jojo/230830_10X_bf_lux_Drew_biospa_Drawer5 30-Aug-2023 16-25-25/230830_Plate 1!PLATE_ID!_'
os.chdir(directory)

data_directory = directory + '/results_data/'
image_directory = directory + '/results_images/'
if os.path.exists(data_directory):
    shutil.rmtree(data_directory)
if os.path.exists(image_directory):
    shutil.rmtree(image_directory)

# Create a new directory
os.mkdir(data_directory)
os.mkdir(image_directory)

# Set your condition name and wells here -- each condition will get a plot
conditions = {
              'Vc_100n' : ['A1_', 'A2_', 'A3_'],
              'Vc_10n' : ['B1_', 'B2_', 'B3_'],
              'Vc_5n' : ['C1_', 'C2_', 'C3_'],
              'Vc_1n' : ['D1_', 'D2_', 'D3_'],
              'Vc_05n' : ['E1_', 'E2_', 'E3_'],
              'Vc_01n' : ['F1_', 'F2_', 'F3_'],
              'Vc_005n' : ['G1_', 'G2_', 'G3_'],
              'Vc_20lys' : ['A4_', 'A5_', 'A6_'],
              'Vc_10lys' : ['B4_', 'B5_', 'B6_'],
              'Vc_5lys' : ['C4_', 'C5_', 'C6_'],
              'Vc_25lys' : ['D4_', 'D5_', 'D6_'],
              'Vc_125lys' : ['E4_', 'E5_', 'E6_'],
              'Vc_0625lys' : ['F4_', 'F5_', 'F6_'],
              'Vc_0lys' : ['G4_', 'G5_', 'G6_'],
              }

# Image parameters
acquisition_frequency = 2 # n times every hour -- change this
bf_sig_output = 2
sig_processing = 7
temporal_median_window = 5
blockRadius = 50
bf_thresh = 1000
bf_thresh2 = 113
img_shift_thresh = 100
dilat_ker = np.ones((5,5), np.uint8)
dilat_ker2 = np.ones((3,3), np.uint8)
min_size = 1100

def filter_connected_components(binary_mask, min_size):
    # Label connected components
    labeled_mask, num_labels = label(binary_mask, connectivity=2, return_num=True)

    # Iterate through unique labels (exclude background which is 0)
    for label_num in range(1, num_labels+1):
        component_size = np.sum(labeled_mask == label_num)

        if component_size < min_size:
            labeled_mask[labeled_mask == label_num] = 0

    # Return binary mask with filtered components
    return labeled_mask > 0

# Pythonic implementation of local contrast normalization
def normalize_local_contrast(img, blockRadius):

    block_size = blockRadius * 2 + 1
    block_average = uniform_filter(img, size=(0, block_size, block_size))
    img_normalized = img - block_average + 37488.0

    return img_normalized

# Function for cropping away boundary artifacts after registration
def crop(arr, val=0):
    mask = np.amin(arr, axis=0) != val
    mask_i = mask.any(axis=1)
    mask_j = mask.any(axis=0)
    slices_i = np.where(mask_i)[0][[0, -1]]
    slices_j = np.where(mask_j)[0][[0, -1]]
    return arr[:, slices_i[0]:slices_i[1]+1, slices_j[0]:slices_j[1]+1]

# Function to register images in a timeseries
def register(timeseries, img_shift_thresh):
    registered = np.zeros_like(timeseries, dtype=timeseries.dtype)
    registered[0] = timeseries[0]

    shifts = np.empty((timeseries.shape[0], 2), dtype=np.float64)
    cum_shift = np.array([0, 0], dtype=np.float64)
    shifts[0] = cum_shift

    # Register each image in the t-direction
    for i in range(1, timeseries.shape[0]):
        # Compute the phase cross-correlation between imepoints to get shift
        img_shift, error, _ = phase_cross_correlation(timeseries[i - 1], timeseries[i],
                                                  upsample_factor=10, normalization=None)
        cum_shift = np.add(cum_shift, img_shift, out=cum_shift)

        if np.linalg.norm(img_shift, 2) < img_shift_thresh:
            # Apply the computed shift
            registered[i] = shift(timeseries[i], shift=cum_shift)
            shifts[i] = cum_shift
        else:
            print("Error in registration")
            registered[i:, :, :] = timeseries[i:, :, :]
            shifts[i:, :] = np.zeros((shifts.shape[0] - i, 2), dtype=np.float64)
            break

    reg_crop = crop(registered)
    return reg_crop, shifts

def temporal_median_filter(images, window_size=temporal_median_window):
    half_window = window_size // 2
    median_filtered_images = []

    for i in range(images.shape[0]):
        if i == 0:
            window = images[:window_size-1]
        elif i == images.shape[0] - 1:
            window = images[-1*(window_size-1):]
        else:
            start = max(0, i - half_window)
            end = min(images.shape[0], i + half_window + 1)
            window = images[start:end]

        median_image = np.median(window, axis=0)
        median_filtered_images.append(median_image)

    return np.asarray(median_filtered_images, dtype=np.uint8)

data = []
num_plots = 0
for j, (_, values) in enumerate(conditions.items()):
    print('Analyzing '+ _ +'...')

    replicates = len(conditions[_])
    extract_lists = replicates

    file_dict = {'Bright': []}
    for well in conditions[_]:

        matching_files = natsorted([f for f in os.listdir() if well in f])
        for file_type in file_dict.keys():
            file_dict[file_type] = [f for f in matching_files if file_type in f]

        BF_images = np.asarray([cv2.imread(f,-1) for f in file_dict['Bright']])
        BF_normalized = normalize_local_contrast(BF_images.astype(np.float32), blockRadius)
        BF_registered, shifts = register(BF_normalized.astype(np.uint16), img_shift_thresh)
        BF_output_images = []
        BF_masks = []
        BF_welldata = [well+'BF']
        img_nobacks = []
        imgs_inverted = []

        for i in range(BF_images.shape[0]):
            # First output images
            img_blurred = cv2.GaussianBlur((BF_registered[i]/256).astype(np.uint8), (0, 0), bf_sig_output)
            BF_output_images.append(img_blurred)

            # Generate noback image
            img = BF_images[i]
            img_inverted = 65535 - img
            imgs_inverted.append(img_inverted)

            img_flattened = img_inverted.flatten()
            img_reshaped = img_flattened.reshape(-1, 1)

            # Fit Bayesian Gaussian mixture model
            gmm = BayesianGaussianMixture(n_components=1, covariance_type='diag')
            gmm.fit(img_reshaped)

            min_mean_idx = np.argmin(gmm.means_)
            means = gmm.means_[min_mean_idx]
            img_noback = img_inverted.astype(np.float32) - means
            img_nobacks.append(img_noback)

        BF_output_images = np.asarray(BF_output_images)
        img_nobacks = np.asarray(img_nobacks)
        img_nobacks = np.asarray([shift(img, shift = cum_shift, mode='constant', cval=-1e6)
                                  for img, cum_shift in zip(img_nobacks, shifts)])
        img_nobacks = crop(img_nobacks, val=-1e6)
        imgs_inverted = np.asarray(imgs_inverted)
        imgs_inverted = np.asarray([shift(img, shift = cum_shift)
                                  for img, cum_shift in zip(imgs_inverted, shifts)])
        imgs_inverted = crop(imgs_inverted)
        if 'Vc' in _ or 'Va' in _:
            img_mask = ((255-BF_output_images) > bf_thresh2).astype(np.uint8)
            img_mask = temporal_median_filter(img_mask)
            img_masks = np.asarray([cv2.dilate(cv2.morphologyEx(cv2.medianBlur(img_mask_slice, 3),cv2.MORPH_CLOSE, dilat_ker), dilat_ker2)
                                    for img_mask_slice in img_mask])

        # Species-specific processing of background subtracted images to generate masks
        unnormalized_masks = []
        normalized_masks = []
        for i in range(img_nobacks.shape[0]):
            if 'Vc' in _ or 'Va' in _:
                # Vibrio cholerae and Vibrio anguillarum
                img_mask = img_masks[i]
                BF_masks.append(img_mask)
            elif 'Vp' in _:
                # Vibrio parahaemolyticus
                normalized_mask = ((255-BF_output_images[i]) > bf_thresh2).astype(np.uint8)
                normalized_mask = cv2.morphologyEx(normalized_mask, cv2.MORPH_OPEN, dilat_ker)
                normalized_mask = cv2.medianBlur(normalized_mask, 3)
                unnormalized_mask = (img_nobacks[i] > bf_thresh).astype(np.uint8)
                unnormalized_mask = cv2.morphologyEx(unnormalized_mask, cv2.MORPH_OPEN, dilat_ker)
                unnormalized_mask = cv2.medianBlur(unnormalized_mask, 5)
                unnormalized_mask = cv2.morphologyEx(unnormalized_mask, cv2.MORPH_CLOSE, dilat_ker)
                normalized_masks.append(normalized_mask)
                unnormalized_masks.append(unnormalized_mask)
            else:
                # Vibrio vulnficius
                img_mask = (img_nobacks[i] > bf_thresh).astype(np.uint8)
                img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, dilat_ker)
                img_mask = cv2.medianBlur(img_mask, 5)
                img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, dilat_ker)
                BF_masks.append(img_mask)

        # Handle V.p. case
        if 'Vp' in _:
            unnormalized_masks = np.asarray(unnormalized_masks)
            normalized_masks = np.asarray(normalized_masks)
            for n in range(1, normalized_masks.shape[0]):
                overlap = np.logical_and(normalized_masks[n-1], unnormalized_masks[n])
                normalized_masks[n][overlap] = 1
                # Dilate and fill holes
                normalized_masks[n] = cv2.erode(normalized_masks[n], dilat_ker)
                normalized_masks[n] = binary_fill_holes(normalized_masks[n])
            BF_masks = normalized_masks

        # Remove dust from masks
        BF_masks = np.asarray(BF_masks, dtype=np.int64)
        initial_nonzero = BF_masks[0] > 0
        transitioned_to_zero = (np.cumprod((BF_masks), axis=0) == 0) & (initial_nonzero)
        frames_to_zero = np.argmax(transitioned_to_zero, axis=0)
        frame_indices = np.arange(BF_masks.shape[0])[:, np.newaxis, np.newaxis]
        dust_mask = (frame_indices < frames_to_zero) & (frames_to_zero != 0)
        BF_masks[dust_mask] = 0
        bf_signal = [np.average(np.where(BF_masks[i] > 0, imgs_inverted[i], 0)) for i in range(imgs_inverted.shape[0])]
        BF_welldata += bf_signal

        # Output images and data
        data.append(BF_welldata)
        tif.imwrite(image_directory + well+"BF_images.tiff",
                    BF_output_images, imagej=True, metadata={'axes' : 'TYX'})

        # Overlay mask on brightfield
        overlay = np.zeros((BF_output_images.shape[0], BF_output_images.shape[1], BF_output_images.shape[2], 3), dtype=np.uint8)
        overlay[..., 0] = BF_output_images
        overlay[..., 1] = BF_output_images
        overlay[..., 2] = BF_output_images
        overlay[..., 0] = np.where(BF_masks, 255, overlay[..., 0])
        # Save the overlay as a TIFF file with metadata TYX
        tif.imwrite(image_directory + well+"BF_mask.tiff",
                    overlay, photometric='rgb')

# Write out data
data = pd.DataFrame(data).transpose()
columns = data.iloc[0]
data = data[1:]
data.columns = columns

data.to_csv(data_directory + 'data.csv', index=False)

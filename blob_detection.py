from scipy import ndimage
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import measure, morphology, segmentation



def detect_blobs(ground_truth: torch.Tensor, predicted: torch.Tensor, threshold:int = 0.3, verbose = False):
    
    structure = np.ones((3, 3, 3), dtype=np.int)
    ground_truth = np.array(ground_truth[0].cpu().detach().numpy(),dtype=np.uint8)
    predicted = np.array(predicted[0].cpu().detach().numpy(),dtype=np.uint8)

    labeled, ncomponents = measure.label(ground_truth, return_num=True)
    dilated_gt = ndimage.binary_dilation(ground_truth,iterations=15)
    dilated_labeled, dilated_ncomponents = measure.label(dilated_gt, return_num=True)
    new_label = np.zeros_like(labeled)
    for n in range(1,dilated_ncomponents+1):
        new_label_indices = (labeled != 0) & (dilated_labeled == n)
        new_label[new_label_indices] = n
    new_label = morphology.remove_small_objects(new_label, min_size = 10)

    blobs_detected = 0

    # print(labeled)

    detected_sizes = []
    undetected_sizes = []
    det_seg_sizes = []
    undet_seg_sizes = []
    if verbose:
        print("--------- Scanning image -------")
    for i in range(1,dilated_ncomponents+1):
        # print("----------- Checking Blob -----------")
        blob = (new_label == i)
        count = np.count_nonzero((predicted != 0) & blob)
        blob_size = np.count_nonzero(blob)
        try:
            proportion_detected = count/blob_size
        except:
            continue
        # if proportion_detected >= threshold:
        if count >= 1:
            if verbose:
                print("Detected blob!")
            blobs_detected += 1
            if blob_size>=1000:
                detected_sizes.append(blob_size)
                det_seg_sizes.append(count)
        else:
            if blob_size>=1000:
                undetected_sizes.append(blob_size)
                undet_seg_sizes.append(count)
            if verbose:
                print("Blob not detected :(")
        if verbose:
            print("Blob details:")
            print("pixels detected: ",count)
            print("proportion detected: ", proportion_detected)
            print("blob size: ", blob_size)

    if verbose:
        print("All blobs scanned")
        print("Detected blobs: ", blobs_detected)
        print("Total number of blobs: ", dilated_ncomponents)
    # _, axs = plt.subplots(5,3)  
    # axs[0,0].imshow(labeled[:,:,50])
    # axs[0,1].imshow(dilated_labeled[:,:,50])
    # axs[0,2].imshow(new_label[:,:,50])
    # axs[1,0].imshow(labeled[:,:,70])
    # axs[1,1].imshow(dilated_labeled[:,:,70])
    # axs[1,2].imshow(new_label[:,:,70])
    # axs[2,0].imshow(labeled[:,:,90])
    # axs[2,1].imshow(dilated_labeled[:,:,90])
    # axs[2,2].imshow(new_label[:,:,90])
    # axs[3,0].imshow(labeled[:,:,110])
    # axs[3,1].imshow(dilated_labeled[:,:,110])
    # axs[3,2].imshow(new_label[:,:,110])
    # axs[4,0].imshow(labeled[:,:,130])
    # axs[4,1].imshow(dilated_labeled[:,:,130])
    # axs[4,2].imshow(new_label[:,:,130])
    # plt.show()

    return detected_sizes, undetected_sizes, det_seg_sizes, undet_seg_sizes

def count_over_thresh(ground_truth: torch.Tensor, predicted: torch.Tensor, threshold:int = 0.3, verbose = False):
    structure = np.ones((3, 3, 3), dtype=np.int)
    ground_truth = np.array(ground_truth[0].cpu().detach().numpy(),dtype=np.uint8)
    predicted = np.array(predicted[0].cpu().detach().numpy(),dtype=np.uint8)

    labeled, ncomponents = measure.label(predicted, return_num=True)
    # dilated = ndimage.binary_dilation(predicted,iterations=15)
    # dilated_labeled, dilated_ncomponents = measure.label(dilated, return_num=True)
    # new_label = np.zeros_like(labeled)
    # for n in range(1,dilated_ncomponents+1):
    #     new_label_indices = (labeled != 0) & (dilated_labeled == n)
    #     new_label[new_label_indices] = n
    # new_label = morphology.remove_small_objects(new_label, min_size = 10)

    segmented_blobs_are_lesions = 0
    num_comps_over_thresh = 0
    for i in range(1,ncomponents+1):
        blob = (labeled == i)
        blob_size = np.count_nonzero(blob)

        if blob_size>=1000:
            num_comps_over_thresh += 1
            count = np.count_nonzero((ground_truth != 0) & blob)
            if count>=1:
                segmented_blobs_are_lesions += 1

    
    return segmented_blobs_are_lesions, num_comps_over_thresh 


def plot_hist(detected, undetected):
    max_blob_size = max(detected + undetected)
    min_blob_size = min(detected + undetected)

    print("Max blobl size = ", max_blob_size)
    print("Min blob size = ", min_blob_size)
    print("Detected " + str(len(detected)) + " out of " + str(len(undetected) + len(detected)))

    # _, axs = plt.subplots(1,2)  
    # axs[0].hist(detected,bins=100, range=[0,1000])
    bins = np.linspace(0, max_blob_size, 100)
    # plt.rc('axes', labelsize=30)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['figure.figsize'] = [7, 5]
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)
    plt.hist(detected, bins=bins, alpha=0.5, label="Detected")
    plt.hist(undetected, bins=bins, alpha=0.5, label="Undetected")
    plt.ylabel("Frequency")
    plt.xlabel("Blob Size (number of voxels)")
    plt.ylim((0,225))
    plt.legend(loc='upper right')
    
    # plt.rcParams.update({'font.size': 62})

    # axs[1].hist(undetected,bins=100, range = [0,1000])
    # axs[1].hist(undetected,bins=100)

    
    detected_percentile_25 = np.percentile(detected, 25)
    detected_percentile_75 = np.percentile(detected, 75)
    try:
        undetected_percentile_25 = np.percentile(undetected, 25)
        undetected_percentile_75 = np.percentile(undetected, 75)
    except:
        undetected_percentile_25 = 0
        undetected_percentile_75 = 0
    full_data_25 = np.percentile(undetected + detected, 25)
    full_data_75 = np.percentile(undetected + detected, 75)

    print("Percentiles")
    print("detected: 25th percentile: ",detected_percentile_25)
    print("detected: 75th percentile: ",detected_percentile_75)
    print("undetected: 25th percentile: ",undetected_percentile_25)
    print("undetected: 75th percentile: ",undetected_percentile_75)
    print("all blobs: 25th percentile: ",full_data_25)
    print("all blobs: 75th percentile: ",full_data_75)

    plt.show()

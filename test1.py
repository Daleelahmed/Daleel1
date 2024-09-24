import os
import nibabel as nib
import numpy as np

root_dir = "/home/daleelahmed1995/kits19/data"
output_dir = "/home/daleelahmed1995/kits19/NiftiOutput"


# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all subdirectories (case folders) in the root directory
case_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

# Initialize a counter to track the file number
file_number = 1

# Specify the case from which you want to restart
start_case = 'case_00203'
skip_previous_cases = True

# Loop over each case folder to load the NIfTI files
for case_folder in case_folders:
    # Skip previous cases until reaching the start_case
    if skip_previous_cases:
        if case_folder == start_case:
            skip_previous_cases = False  # Stop skipping once we reach the start case
        else:
            continue  # Skip the folder if it's before the start case
    
    image_file = os.path.join(root_dir, case_folder, 'imaging.nii.gz')
    mask_file = os.path.join(root_dir, case_folder, 'segmentation.nii.gz')
    
    if os.path.exists(image_file) and os.path.exists(mask_file):
        print(f"Processing file number: {file_number} for case folder: {case_folder}")

        img_nii = nib.load(image_file)
        mask_nii = nib.load(mask_file)

        img_dataobj = img_nii.dataobj
        mask_dataobj = mask_nii.dataobj

        # Compute dynamic min/max values for the current case's image and mask
        dynamic_min_img = np.min(img_dataobj)
        dynamic_max_img = np.max(img_dataobj)
        dynamic_min_mask = np.min(mask_dataobj)
        dynamic_max_mask = np.max(mask_dataobj)

        print("Dynamic Min value of image:", dynamic_min_img)
        print("Dynamic Max value of image:", dynamic_max_img)
        print("Shape of the image:", img_dataobj.shape)
        
        case_output_dir = os.path.join(output_dir, case_folder)
        if not os.path.exists(case_output_dir):
            os.makedirs(case_output_dir)
        
        # Process each slice in the image data
        for slice_index in range(img_dataobj.shape[0]):
            slice_img = img_dataobj[slice_index, :, :]  # Extract each slice
            slice_img_normalized = (slice_img - dynamic_min_img) / (dynamic_max_img - dynamic_min_img)
            print(f"Image Slice {slice_index}: Min={np.min(slice_img_normalized)}, Max={np.max(slice_img_normalized)}, Shape={slice_img_normalized.shape}")
            
            normalized_img_nii = nib.Nifti1Image(slice_img_normalized, img_nii.affine, img_nii.header)
            nib.save(normalized_img_nii, os.path.join(case_output_dir, f'Normalized_Imaging_Slice_{slice_index}.nii.gz'))

        # Process each mask slice
        for slice_index in range(mask_dataobj.shape[0]):
            slice_mask = mask_dataobj[slice_index, :, :]  # Extract each slice
            slice_mask_normalized = (slice_mask - dynamic_min_mask) / (dynamic_max_mask - dynamic_min_mask)
            print(f"Mask Slice {slice_index}: Min={np.min(slice_mask_normalized)}, Max={np.max(slice_mask_normalized)}, Shape={slice_mask_normalized.shape}")

            normalized_mask_nii = nib.Nifti1Image(slice_mask_normalized, mask_nii.affine, mask_nii.header)
            nib.save(normalized_mask_nii, os.path.join(case_output_dir, f'Normalized_Segmentation_Slice_{slice_index}.nii.gz'))

        # Increment the file number after processing each case
        file_number += 1

    elif os.path.exists(image_file) and not os.path.exists(mask_file):
        print(f"Mask file missing for case: {case_folder}")

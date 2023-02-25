import os
import subprocess

# Whereis bet :
FILEPATH_TO_BET_FSL = "/home/shared_space/software/fsl/bin/bet"

# ---- Example commandline usage of BRAINSFit from 3D Slicer (but use the main function, because it calls it with better parameters) ---
# bet /path/of/original/T1.nii.gz /path/where/to/to/save/output/T1_brain.nii.gz -m -R
# Output:
# same_name.nii.gz
# _mask.nii.gz


def extract_skull_fsl(fpath_img_orig,
                      fpath_img_out,
                      fpath_brainmask_out):

    command = FILEPATH_TO_BET_FSL + \
                    " " + fpath_img_orig + \
                    " " + fpath_img_out + \
                    " -m" + \
                    " -R" + \
                    " -f 0.4" + \
                    " -g 0" + \
                    " -v"  # Verbose
    subprocess.call(command, shell="True")

    # rename outputs
    fpath_img_out_no_type = fpath_img_out[:fpath_img_out.rfind(".nii.gz")]
    fpath_brainmask_old = fpath_img_out_no_type + "_mask.nii.gz"
    os.rename(fpath_brainmask_old, fpath_brainmask_out)


def extract_skull(case,
                  fpath_img_orig,
                  fpath_img_out,
                  fpath_brainmask_out,
                  software="fsl"):

    print("------- Starting main process for case: ", case, " ----------")
    # Create output folder
    if not os.path.exists(os.path.dirname(fpath_img_out)):
        os.mkdir(os.path.dirname(fpath_img_out))

    if software == "fsl":
        extract_skull_fsl(fpath_img_orig, fpath_img_out, fpath_brainmask_out)
    elif software == "robex":
        pass
    else:
        raise NotImplementedError()


# Example read file path: /home/shared_space/data/ATLAS_R2.0/ATLAS_2/Training/R001/sub-r001s001/ses-1/anat/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
# Example write file path brain: /home/sedm6251/projectMaterial/skullStripped/ATLAS/images/brain.nii.gz"
# Example write file path brain: /home/sedm6251/projectMaterial/skullStripped/ATLAS/masks/brain_mask.nii.gz"

def main():
    # Input
    # Brain image with skull
    main_folder_orig = "/home/shared_space/data/ATLAS_R2.0/ATLAS_2/Training/R001"
    subfolder_orig = "ses-1/anat"  # Adjust this to your own folder structure.
    fname_suffix_img_orig = "_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz" # Adjust this to the name of your images

    # Output
    main_folder_out = "/home/sedm6251/projectMaterial/skullStripped/ATLAS"
    # subfolder_out = ""  # May not exist

    # fname_suffix_img_out = "-T1w_brain.nii.gz"
    # fname_suffix_brainmask_out = "-T1w-brainmask.nii.gz"
    fname_suffix_img_out = ""
    fname_suffix_brainmask_out = ""

    # Logic
    cases = os.listdir(main_folder_orig)
    cases.sort()
    for case in cases:
        if not os.path.isdir(os.path.join(main_folder_orig, case)):
            continue  # In case of subfiles instead of subfolders, e.g. readme etc.

        # Input
        case_folder_orig = os.path.join(main_folder_orig, case, subfolder_orig)
        fname_img_orig = case + fname_suffix_img_orig
        fpath_img_orig = os.path.join(case_folder_orig, fname_img_orig)
        # fpath_img_orig = "/home/shared_space/data/ATLAS_R2.0/ATLAS_2/Training/R001/sub-r001s001/ses-1/anat/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"


        # Output
        case_folder_out = os.path.join(main_folder_out, case)
        fname_img_out = case + fname_suffix_img_out
        fpath_img_out = os.path.join(case_folder_out, fname_img_out)
        fname_brainmask_out = case + fname_suffix_brainmask_out
        fpath_brainmask_out = os.path.join(case_folder_out, fname_brainmask_out)

        print("fpath_img_orig=", fpath_img_orig)
        print("fpath_img_out=", fpath_img_out)
        print("fpath_brainmask_out=", fpath_brainmask_out)

        extract_skull(case,
                      fpath_img_orig,
                      fpath_img_out,
                      fpath_brainmask_out,
                      "fsl")


if __name__ == "__main__":
    main()


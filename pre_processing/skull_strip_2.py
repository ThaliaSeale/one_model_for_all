#Robex is using T1 only to extract the brain. Here I do it:
import os
import subprocess
import shutil



def stringIsNumber(s) :
    try :
        int(s)
        return True
    except ValueError:
        return False

def runRobexOnSingleT1ImageAndSaveTo(filenameAndPathToT1,
                                     filenameWithPathToSaveBrain,
                                     filenameWithPathToSaveMask) :
    FILEPATH_TO_ROBEX = "/home/shared_space/software/robex/ROBEX/runROBEX.sh"
    subprocess.call(FILEPATH_TO_ROBEX + " " + filenameAndPathToT1 + " " + filenameWithPathToSaveBrain + "  " + filenameWithPathToSaveMask,
                    shell=True)
        
    
def main():
    # Check that FILEPATH_TO_ROBEX is accessible and executable for you first (try to just run that .sh script from your terminal.
    
    # Example read file path: /home/shared_space/data/ATLAS_R2.0/ATLAS_2/Training/R001/sub-r001s001/ses-1/anat/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
    # Example write file path brain: /home/sedm6251/projectMaterial/skullStripped/ATLAS/images/brain.nii.gz"
    # Example write file path brain: /home/sedm6251/projectMaterial/skullStripped/ATLAS/masks/brain_mask.nii.gz"
    training_set_path = "/home/shared_space/data/ATLAS_R2.0/ATLAS_2/Training/"
    subfolder_path = "ses-1/anat/"

    subsets = os.listdir(training_set_path)
    subsets.sort()
    # print("Found subsets:")
    # print(subsets)
    for subset in subsets:
        # e.g. subsets = [R001, R002,...]
        if not os.path.isdir(os.path.join(training_set_path, subset)):
            # print("Subset path is not a directory")
            continue  # In case of subfiles instead of subfolders, e.g. readme etc.
        
        # subset path is e.g. /home/shared_space/data/ATLAS_R2.0/ATLAS_2/Training/R001/
        subset_path = os.path.join(training_set_path, subset)
        cases = os.listdir(subset_path)
        cases.sort()
        # print("Found cases:")
        # print(cases)

        #  cases = [sub-r001s001, sub-r001,s002,...]
        for case in cases:
            if not os.path.isdir(os.path.join(subset_path, case)):
                # print("Case is not a directory")
                continue  # In case of subfiles instead of subfolders, e.g. readme etc.

            #  join on the ses-1/anat part of the directory
            file_parent_folder_path = os.path.join(subset_path, case,subfolder_path)
            file_name = case + "_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz" 
            file_path = os.path.join(file_parent_folder_path,file_name)
            # print("Have ended with file path = " + file_path)
            
            label_file_name = case + "_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
            label_path = os.path.join(file_parent_folder_path, label_file_name)

            save_name_image = case + "_img.nii.gz"
            save_path_image = os.path.join("/home/sedm6251/projectMaterial/skullStripped/ATLAS/images/",save_name_image)
            
            save_name_mask = case + "_mask.nii.gz"
            save_path_mask = os.path.join("/home/sedm6251/projectMaterial/skullStripped/ATLAS/masks/",save_name_mask)

            save_name_label = case + "_label.nii.gz"
            save_path_label = os.path.join("/home/sedm6251/projectMaterial/skullStripped/ATLAS/labels/",save_name_label)

            shutil.copy(label_path,save_path_label)

            
            print("File paths:")
            print(file_path)
            print(save_path_image)
            print(save_path_mask)

    
            # runRobexOnSingleT1ImageAndSaveTo("/home/shared_space/data/ATLAS_R2.0/ATLAS_2/Training/R001/sub-r001s001/ses-1/anat/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz", "/home/sedm6251/projectMaterial/skullStripped/ATLAS/brain.nii.gz", "/home/sedm6251/projectMaterial/skullStripped/ATLAS/brainmask.nii.gz")
            # runRobexOnSingleT1ImageAndSaveTo(str(file_path),str(save_path_image),str(save_path_mask))
main()



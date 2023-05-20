import os
import glob
from test_network_with_random_missing_modalities import test
import utils
import torch
from monai.utils import set_determinism
from test_all import Net

dir = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/"
files = os.listdir(dir)
files.sort()


def check_for_pre_or_number(folder):
    files_found = []
    files_in_folder = os.listdir(folder)
    files_in_folder.sort()
    for file in files_in_folder:
        path = os.path.join(folder, file)
        # print(path)
        if file == "runs":
            continue
        if os.path.isdir(path):
            # print("is dir")
            files_found = files_found + check_for_pre_or_number(path)
        else:
            if ("20" in file or "50" in file) and ".pth" in file and "BEST" in file:
                if "ISLES" not in file and "ATLAS_50" not in file and "HEM" not in file:
                    files_found.append(path)
                # print(file)
    else:
        return files_found

if __name__ == "__main__":
    files = check_for_pre_or_number(dir)

    device_id = 0
    cuda_id = "cuda:" + str(device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    for f in files:
        net_type = ""
        parent_folder_index = f.find("TRAIN")
        parent_folder_index_end = f.find("/", parent_folder_index)
        parent_folder_name = f[parent_folder_index: parent_folder_index_end]
        net_name = f[parent_folder_index_end]
        if "UNET" in f:
            net_type = "UNet"
        elif "MSFN" in f:
            net_type = "MSFN"
        
        datasets = ["BRATS", "ATLAS", "MSSEG", "ISLES2015", "TBI", "WMH"]
        dataset_modalities = [[0,1,2,3],[0],[0,1,2,3,4],[0,1,2,3],[0,1,2,3],[0,1]]
        unet_dataset_order = [[0,1,2,3], [0],[0,2,3,4,1], [1,2,3,0], [0,2,3,1],[0,1]]
        for i, dataset in enumerate(datasets):
            if dataset in parent_folder_name:
                mods = dataset_modalities[i]
                unet_order = unet_dataset_order[i]
                data = dataset
                break

        out = "Net(\"" + f + "\", \"" + net_type + "\", 0, { \"" + data + " \" :" + str(unet_order) + "} ),"
        out.replace("\\", "")
        print(out)


        # try:
        #     net = Net(f, net_type, 5, channel_map=)

        # print("*************** TESTING NET " + str(net.file_path) + " **************")


        # model = utils.create_net(net, device, cuda_id)

        # for i, dataset in enumerate(datasets):
        #     print("************** TESTING DATSET " + dataset + " ***************")
        #     dataloader = utils.create_dataset(dataset)
        #     modalities = utils.create_modality_combinations(dataset_modalities[i])
        #     # REMOVE THIS
        #     # modalities = [(0,1,2)]
        #     for combination in modalities:
        #         # print(combination)
        #         test.test(model,
        #             dataloader,
        #             dataset,
        #             combination,
        #             net,
        #             device,
        #             dataset_modalities[i],
        #             save_outputs=False,
        #             save_path="/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TEST_TBI/SWI_tests/UNET_BRATS_ATLAS_MSSEG_WMH/no_SWI_",
        #             detect_blobs = False)
        #             # save_path="/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TEST_MSSEG/UNET_All_modalities/UNET_All_modalities_")






        




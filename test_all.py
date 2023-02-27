import test_network_with_random_missing_modalities as test
import torch
from monai.utils import set_determinism

class Net:
    def __init__(self, file_path:str, net_type:str, modalities_trained_on: int, channel_map:dict):
        self.file_path = file_path
        self.net_type = net_type
        self.modalities_trained_on = modalities_trained_on
        self.channel_map = channel_map

if __name__=="__main__":

    nets = [

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_MSSEG_RAND_BEST_MSSEG.pth", "HEM spatial attention", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/UNET/RAND/UNET_MSSEG_RAND_BEST_MSSEG.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_ATLAS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_ISLES.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_Epoch_599.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_ATLAS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_ISLES.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_Epoch_449.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_ATLAS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_ISLES.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_Epoch_449.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN/MSFN_BRATS_ATLAS_ISLES_RAND_BEST_ATLAS.pth", "MSFN", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN/MSFN_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth", "MSFN", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN/MSFN_BRATS_ATLAS_ISLES_RAND_BEST_ISLES.pth", "MSFN", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN/MSFN_BRATS_ATLAS_ISLES_RAND_Epoch_999.pth", "MSFN", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_Epoch_449.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_Epoch_499.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_ATLAS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_MSSEG.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_Epoch_449.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_Epoch_999.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_ISLES.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_MSSEG.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_Epoch_449.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_BRATS.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_ISLES.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_MSSEG.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_Epoch_449.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_ATLAS.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_ISLES.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_MSSEG.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_Epoch_999.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]})

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/MSFN_PAIRED_BRATS_ALL_Epoch_199.pth", "MSFN",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]})

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_Epoch_499.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_ISLES_BEST_ATLAS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_ISLES_BEST_BRATS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_ISLES_Epoch_449.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_BEST_ATLAS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_BEST_BRATS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_Epoch_499.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),


    ]

    device_id = 0
    cuda_id = "cuda:" + str(device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    set_determinism(seed=0)
    print("determinism seed = 0")

    datasets = ["BRATS", "ATLAS", "MSSEG", "ISLES2015"]
    #REMOVE THIS
    datasets = ["ISLES2015"]

    dataset_modalities = [[0,1,2,3],[0],[0,1,2,3,4],[0,1,2,3]]
    # REMOVE THIS
    dataset_modalities = [[0,1,2,3]]

    for net in nets:
        print("*************** TESTING NET " + net.file_path + " **************")
        model = test.create_net(net.file_path, net.net_type, device, net.modalities_trained_on,cuda_id)
        model.eval()
        for i, dataset in enumerate(datasets):
            print("************** TESTING DATSET " + dataset + " ***************")
            dataloader = test.create_dataset(dataset)
            modalities = test.create_modality_combinations(dataset_modalities[i])
            # REMOVE THIS
            modalities = [(0,1,2,3)]
            for combination in modalities:
                # print(combination)
                test.test(model,
                    dataloader,
                    combination,
                    device,
                    net.channel_map[dataset],
                    dataset_modalities[i],
                    net.net_type,
                    net.modalities_trained_on,
                    save_outputs=False,
                    save_path="/home/sedm6251/projectMaterial/baseline_models/Combined_Training/Test_ISLES/MSFNP_BRATS_ATLAS_MSSEG_w_DWI/MSFNP_DWI_")
                    # save_path="/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TEST_MSSEG/UNET_All_modalities/UNET_All_modalities_")


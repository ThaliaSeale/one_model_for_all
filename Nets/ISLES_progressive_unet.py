from Nets.theory_UNET_progressive import theory_UNET, theory_UNET_progressive

import torch
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def extract_features(input_data,layers):
    out = pretrained_model(input_data)
    pretrained_model_features = list()
    for layer in layers:
        pretrained_model_features.append(features[layer])
    return pretrained_model_features

cuda_id = "cuda:0" # this needs to be changed when running the model
# cuda_id = "cuda:1" # this needs to be changed when running the model
device = torch.device(cuda_id)
torch.cuda.set_device(cuda_id)
    
# pretrained_model_path = "results/23_06__14_26_exc_WMH/models/23_06__14_26_exc_WMH23_06__14_26_exc_WMH_Epoch_549.pth" 
pretrained_model_path = "Base_model/For_Finetune_ISLES.pth"
print("LOADING PRETRAINED MODEL:", pretrained_model_path)

manual_channel_map = [1,3,5,6] # not sure if I did this correctly
modalities_when_trained =  ['DP', 'FLAIR', 'SWI', 'T1', 'T1c', 'T2']
total_modalities = modalities_when_trained

pretrained_model = theory_UNET(in_channels = len(modalities_when_trained),
                                out_channels=1).to(device)
pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
pretrained_model.eval()
for param in pretrained_model.parameters():
    param.requires_grad = False

# registering forward hooks for the new model to get the activations
layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'up_stage_1', 'up_stage_2', 'up_stage_3', 'up_stage_4']
for layer in layers:
    getattr(pretrained_model,layer).register_forward_hook(get_features(layer))
features = {} # placeholder for the features

class ISLES_progressive_UNET(theory_UNET_progressive):

    def __init__(self,
            in_channels: int,
            out_channels:int = 1,
            last_layer_conv_only:bool = True
        ) -> None:
        super().__init__(in_channels,out_channels,last_layer_conv_only)

        self.pretrained_model = pretrained_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pretrained_input = x[:,0:6,:,:,:] # first 6 modalities were seen, there rest weren't
        pretrained_features = extract_features(pretrained_input,layers) 
        x = x[:,manual_channel_map,:,:,:] # extracting only non-empty modalities
        return super().forward(x,pretrained_features)

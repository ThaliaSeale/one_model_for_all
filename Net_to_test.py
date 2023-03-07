import utils
class Net:
    def __init__(self, file_path:str, net_type:str, modalities_trained_on: int, channel_map:dict, device=None, cuda_id=None):
        self.ensemble = False
        self.file_path = file_path
        self.net_type = net_type
        self.modalities_trained_on = modalities_trained_on
        self.channel_map = channel_map

 

class EnsembleNet:
    def __init__(self, nets: list):
        self.ensemble = True
        # self.nets = nets
        self.net_descriptors = nets
        self.file_path = [net.file_path for net in nets]
    def init_nets(self,device, cuda_id):
        _nets = []
        for net in self.net_descriptors:
            _nets.append(utils.create_net(net,device,cuda_id))
        self.nets = _nets
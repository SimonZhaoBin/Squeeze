def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'squeezenet':
        from .squeezenet import SQUEEZENET
        return SQUEEZENET
    elif network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return d
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))

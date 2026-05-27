import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS device (Apple Silicon)')
    else:
        device = torch.device('cpu')
        print('Using CPU device')
    return device

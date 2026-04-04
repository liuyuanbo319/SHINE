import numpy as np
import torch
from scipy import signal

data_type_dict = {'LEARNED': '_LEARNED.npz', 'NTS': '_NTS.npz', 'REPRESENT': '_REPRE.npz', 'ORIGINAL': '.npz',
                  'NOISE': '_NOISE.npz', 'JITTER': '_JITTER.npz', 'SCALE': '_SCALE.npz', 'SHIFT': '_SHIFT.npz',
                  'NOISE05': '_NOISE05.npz',
                  'NOISE01': '_NOISE01.npz', 'NOISE001': '_NOISE001.npz', 'NOISE0001': '_NOISE0001.npz',
                  'BP05-30': '_05-30.npz', 'BP1-30': '_1-30.npz', 'BP1-25': '_1-25.npz',
                  'IN01': '_IN01.npz', 'IN001': '_IN001.npz', 'IN0001': '_IN0001.npz',
                  'D6': '_D6.npz', 'D4': '_D4.npz',
                  'NORM': '_NORM.npz'}

z_dims = {'ECG200': 12, 'nmt': 96, 'mit-bih-arrhythmia': 48, 'PTBXL_D': 96, 'tuab': 96}


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def bf(sample_freq, low, high, data):
    fs = sample_freq
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b = signal.firwin(5, [low, high], pass_zero='bandpass').astype(np.float32)

    filtered_signal = signal.lfilter(b, [1], data.astype(np.float32)).astype(np.float32)

    # b, a = signal.butter(3, [low, high],btype='bandpass',fs=sample_freq)
    # filtered_signal = signal.filtfilt(b, a, data)

    return filtered_signal


def shift(data, sigma=0.5, IN=False):
    # noise = (np.random.randn(data.shape[0],data.shape[1],data.shape[2]) * sigma).astype(np.float32)
    noise = (np.random.normal(0, sigma, data.shape)).astype(np.float32)
    if IN:
        noise = bf(1280, 1e-6, 1, noise).astype(np.float32)
    return data + noise

def mask(data,mask_ratio = 0.1):
    if np.ndim(data)==2:
        data=data[:,np.newaxis,:]
    B, F, T = data.shape

    num_elements = B * F * T
    num_masked = int(num_elements * mask_ratio)

    mask = np.ones(num_elements, dtype=np.float32)


    masked_indices = np.random.choice(num_elements, num_masked, replace=False)

    mask[masked_indices] = 0


    mask = mask.reshape(B, F, T)


    masked_data = data * mask
    # masked_data = np.squeeze(masked_data)
    return masked_data.astype(np.float32)

def weak_shift(features):
    r = np.random.randint(0, 5, 1)
    if r == 0:
        weak_shift_data = torch.round(torch.stack(features), decimals=6)
    elif r == 1:
        weak_shift_data = torch.round(torch.stack(features), decimals=4)
    elif r == 2:
        weak_shift_data = torch.from_numpy(bf(1280, 0.5, 30, np.stack(features)))
    elif r == 3:
        weak_shift_data = torch.from_numpy(bf(1280, 1, 30, np.stack(features)))
    else:
        weak_shift_data = torch.from_numpy(bf(1280, 1, 25, np.stack(features)))
    return weak_shift_data
def strong_shift(data, sigma=0.1):
    data = torch.stack(data)
    r = np.random.randint(0, 2, 1)
    IN = bool(r)
    noise = (np.random.normal(0, sigma, data.shape)).astype(np.float32)
    if IN:
        noise = bf(1280, 1e-6, 1, noise)
    data = data + torch.from_numpy(noise.astype(np.float32)).to(data.device)
    return data



def sw_shift(data, std=0.1, sw='ws'):
    features, labels, _ = zip(*data)
    if sw == 'ws':
        weak_shift_data = weak_shift(features)
        strong_shift_data = strong_shift(features,std)
        return weak_shift_data, strong_shift_data, torch.stack(labels, dim=0)
    elif sw == 'ss':
        strong_shift_data1 = strong_shift(features, std)
        strong_shift_data2 = strong_shift(features, std)
        return strong_shift_data1, strong_shift_data2, torch.stack(labels, dim=0)
    elif sw == 'os':
        weak_shift_data = weak_shift(features)
        return torch.stack(features), weak_shift_data, torch.stack(labels, dim=0)
    elif sw == 'ww':
        weak_shift_data1 = weak_shift(features)
        weak_shift_data2 = weak_shift(features)
        return weak_shift_data1, weak_shift_data2, torch.stack(labels, dim=0)
    elif sw == 'sw':
        weak_shift_data = weak_shift(features)
        strong_shift_data = strong_shift(features, std)
        return strong_shift_data, weak_shift_data, torch.stack(labels, dim=0)
    else:
        return torch.stack(features), torch.stack(features), torch.stack(labels, dim=0)


def get_coef(model, data):
    model.eval()
    with torch.no_grad():
        T_coef = model.trend_decoder(model.trend_encoder(model.backbone(data)))
        S_coef = model.seasonal_decoder(model.seasonal_encoder(model.backbone(data)))
        trend = T_coef.matmul(model.T)
        seasonality = S_coef.matmul(model.S)
        print('reco loss', torch.nn.functional.mse_loss(trend + seasonality, data))
    return T_coef.cpu().numpy(), S_coef.cpu().numpy(), trend.cpu().numpy(), seasonality.cpu().numpy()

def get_coefs(model,dataloader,k):
    model.eval()
    S_coefs = []
    trends = []
    seasonalitys = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to('cuda')
            T_coef = model.trend_decoder(model.trend_encoder(model.backbone(x)))
            S_coef = model.seasonal_decoder(model.seasonal_encoder(model.backbone(x)))
            trend = T_coef.matmul(model.T)
            seasonality = S_coef.matmul(model.S)
            S_coefs.append(S_coef)
            trends.append(trend)
            seasonalitys.append(seasonality)
    S_coefs= torch.concat(S_coefs,0).cpu().numpy()
    trends= torch.concat(trends,0).cpu().numpy()
    seasonalitys= torch.concat(seasonalitys,0).cpu().numpy()
    np.savez(f'SHINE-{k}K.npz',s_coefs=S_coefs,t=trends,s=seasonalitys)
            # print('reco loss', torch.nn.functional.mse_loss(trend + seasonality, data))

def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels,_ = zip(*data)
    # features = features.perumte(0,2,1)
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1]) # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks
def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
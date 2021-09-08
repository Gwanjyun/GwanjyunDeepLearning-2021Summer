def _nms(hm):
    pool = nn.MaxPool2d(3,1,1)
    hmax = pool(hm)
    keep = (hm == hmax).float()
    return hm * keep

def _bbox_clamp(bbox, imsize):
    bbox[:,[0,2]] = torch.clamp(bbox[:,[0,2]], imsize[1])
    bbox[:,[1,3]] = torch.clamp(bbox[:,[1,3]], imsize[0])
    return bbox 

def gauss2D(hm_index, gt_hm_cpoint, sigma = 10):
    gauss_kernel = (-(hm_index - gt_hm_cpoint).pow(2).sum(-1)/(2*sigma)).exp()
    return gauss_kernel
def scale_shift_pred_depth(pred, gt):
    gt=gt[np.newaxis,:,:]
    pred=pred[np.newaxis,:,:]
    c, h, w = pred.shape
    pred=pred.type(torch.float32)
    gt=gt.type(torch.float32)
    mask_value=1e-3
    max_threshold=80
    mask = (gt > mask_value) & (gt < max_threshold)  # [ c, h, w]
    EPS = 1e-6 * torch.eye(2, dtype=pred.dtype, device=pred.device)
    ones_img = torch.ones((1, h, w), dtype=pred.dtype, device=pred.device)

    pred_valid_mask = pred[mask]
    ones_mask = ones_img[mask]
    pred_valid_ones = torch.stack((pred_valid_mask, ones_mask), dim=0).type(torch.float32)  # [c+1, h, w]
    A_i = torch.matmul(pred_valid_ones, pred_valid_ones.permute(1, 0))  # [2, 2]
    A_inverse = torch.inverse(A_i + EPS)
    gt_mask = gt[mask]
    B_i = torch.matmul(pred_valid_ones, gt_mask)[:, None]  # [2, 1]
    scale_shift_i = torch.matmul(A_inverse, B_i)  # [2, 1]

    ones = torch.ones_like(pred)
    pred_ones = torch.cat((pred, ones), dim=0)  # [2, h, w]
    pred_scale_shift = torch.matmul(pred_ones.permute(1,2,0).reshape(h * w, 2), scale_shift_i)  # [h*w, 1]
    pred_scale_shift = pred_scale_shift.permute(1,0).reshape((h, w))
    return pred_scale_shift

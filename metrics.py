import torch
from pytorch_msssim import ssim, ms_ssim
import lpips
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
    )
    _lpips_model = lpips.LPIPS(net="vgg")


def calculate_psnr(img1, img2, max_val=1.0):
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)

    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))

    psnr = 10 * torch.log10((max_val**2) / mse)

    return psnr.mean()


def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0)


def calculate_ms_ssim(img1, img2):
    return ms_ssim(img1, img2, data_range=1.0)


def calculate_lpips(img1, img2, device="cuda"):
    model = _lpips_model.to(device)
    with torch.no_grad():
        value = 0
        for i in range(img1.shape[0]):
            value += model(img1[i], img2[i])
        return value / img1.shape[0]


def compute_metrics(img1, img2, device="cuda"):
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    ms_ssim = calculate_ms_ssim(img1, img2)
    lpips = calculate_lpips(img1, img2, device)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
    }

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
import os
import csv
from PIL import Image
from config import BATCH_SIZE
from models.model import E2EImageCommunicator, E2E_Channel, E2E_Decoder, E2E_Encoder, E2E_AutoEncoder
from models.qam_model import QAMModem
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


device = torch.device('cuda')

def im_batch_to_image(batch_images):
    '''
    Converts batch of images to a single large image.
    '''
    batch, c, h, w = batch_images.size()

    divisor = int(batch ** 0.5)

    while batch % divisor != 0:
        divisor -= 1

    image = batch_images.view(divisor, batch // divisor, c, h, w)
    image = image.permute(0, 3, 1, 4, 2).contiguous()
    image = image.view(divisor * h, -1, c)
    return image

test_set = datasets.CIFAR10(root="utils/dataset", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)



best_model = 'weights/epoch_141.pth'


QAM_ORDER = 256


loss_function = nn.MSELoss()

if not os.path.isdir('./results'):
    os.mkdir('./results')

for channelname in ['Rayleigh', 'AWGN']:
    if not os.path.isdir(f'./results/{channelname}/'):
        os.mkdir(f'./results/{channelname}/')

    with open(f'./results/{channelname.lower()}_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR','PropSSIM', 'PropMSE', 'PropPSNR', 'QAMSSIM', 'QAMMSE', 'QAMPSNR'])

        for EVAL_SNRDB in range(0, 45, 5):
            qam_modem = QAMModem(snrdB=EVAL_SNRDB, order=QAM_ORDER, channel=channelname)
            model = E2EImageCommunicator(snrdB=EVAL_SNRDB, channel=channelname).to(device)
            model.load_state_dict(torch.load(best_model))
            model.eval()

            ssim_props = 0
            ssim_qams = 0
            mse_props = 0
            mse_qams = 0
            psnr_props = 0
            psnr_qams = 0

            with torch.no_grad():
                for i, (images, _) in enumerate(test_loader):
                    images = images.to(device)
                    int16_images = (images * 255).type(torch.int16)
                    qam_results = torch.tensor(qam_modem(int16_images), dtype=torch.float32).to(device) / 255
                    prop_results = model(images)

                    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
                    prop_results_np = prop_results.cpu().numpy().transpose(0, 2, 3, 1)
                    qam_results_np = qam_results.cpu().numpy().transpose(0, 2, 3, 1)

                    for j in range(images.size(0)):
                        ssim_props += ssim(images_np[j], prop_results_np[j], data_range=1.0, channel_axis = 2, win_size = 3)
                        ssim_qams += ssim(images_np[j], qam_results_np[j], data_range=1.0, channel_axis = 2, win_size = 3)
                        mse_props += loss_function(images[j], prop_results[j]).item()
                        mse_qams += loss_function(images[j], qam_results[j]).item()
                        psnr_props += psnr(images_np[j], prop_results_np[j], data_range=1.0)
                        psnr_qams += psnr(images_np[j], qam_results_np[j], data_range=1.0)

                    if i == 0:
                        img = im_batch_to_image(images.cpu())
                        prop_img = im_batch_to_image(prop_results.cpu())
                        qam_img = im_batch_to_image(qam_results.cpu())

                        Image.fromarray((img.numpy() * 255).astype(np.uint8)).save(f'./results/{channelname}/original_SNR{EVAL_SNRDB}.png')
                        Image.fromarray((prop_img.numpy() * 255).astype(np.uint8)).save(f'./results/{channelname}/proposed_SNR{EVAL_SNRDB}.png')
                        Image.fromarray((qam_img.numpy() * 255).astype(np.uint8)).save(f'./results/{channelname}/256qam_SNR{EVAL_SNRDB}.png')

                    if i == 10:
                        break

            total_images = (i + 1) * BATCH_SIZE
            ssim_props /= total_images
            ssim_qams /= total_images
            mse_props /= total_images
            mse_qams /= total_images
            psnr_props /= total_images
            psnr_qams /= total_images

            print(f'Channel: {channelname} / SNR: {EVAL_SNRDB}dB =======================================')
            print(f'SSIM: (Proposed){ssim_props:.6f} vs. (QAM){ssim_qams:.6f}')
            print(f'MSE:  (Proposed){mse_props:.6f} vs. (QAM){mse_qams:.6f}')
            print(f'PSNR:  (Proposed){psnr_props:.6f} vs. (QAM){psnr_qams:.6f}')

            writer.writerow([EVAL_SNRDB, float(ssim_props), float(mse_props), float(psnr_props), float(ssim_qams), float(mse_qams), float(psnr_qams)])

# # Save intermediate layer images
# with torch.no_grad():
#     images, _ = next(iter(test_loader))
#     images = images.to(device)

#     for model_class in [E2EImageCommunicator, E2E_Encoder, E2E_Channel, E2E_Decoder, E2E_AutoEncoder]:
#         model = model_class(filters=[32, 64, 128], snrdB=10, channel='Rayleigh').to(device)
#         model.load_state_dict(torch.load(best_model))
#         result = model(images[:1])
#         output = result
#         if model_class.__name__ != 'E2EImageCommunicator':
#             # For intermediate layers, flatten channels and normalize to grayscale images
#             output = output / output.max(dim=-1, keepdim=True)[0]
#             output = output.permute(3, 0, 1, 2).contiguous()
#             output = output.view(-1, 32, 1)
#         else:
#             output = output.view(-1, 32, 3)
#         Image.fromarray((output.cpu().numpy() * 255).astype(np.uint8)).save(f'./results/{channelname}/{model_class.__name__}_SNR10.png')

#         if model_class.__name__ == 'E2E_Encoder':
#             result.view(-1).cpu().numpy().tofile(f'./results/{channelname}/constellation.bin')

#     Image.fromarray((images[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(f'./results/{channelname}/E2E_before_SNR10.png')


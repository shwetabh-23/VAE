import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from vae import VAE
import yaml
import torch
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import cv2
import lpips
from scipy.linalg import sqrtm
from numpy import cov, trace, iscomplexobj
from skimage.transform import resize

from torchvision.transforms import Resize
import torchvision.models as models

from PIL import Image
import shutil

from pytorch_fid import fid_score


with open('config.yaml', 'r') as f:
    file = yaml.safe_load(f)

os.makedirs('data', exist_ok= True)
model = VAE(file= file)

inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.eval()

# Define a transformation to preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 expects input images to be 299x299
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to the ImageNet mean and standard deviation
])


bottleneck_layer_size = '256'
model_path = f'/home/harsh/AI-Projects/VAE/models/cifar/size_variation/{bottleneck_layer_size}/model.pth'
#breakpoint()
logger = TensorBoardLogger(model_path[:-9] + 'logs/', name= 'modelv1')
pca = PCA(n_components= 2)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict=state_dict)

def calc_psnr():
    
    batch, y = next(iter(model.testdataloader))
    psnr = []
    for image in batch:
        image1 = get_image_ready(image=image*255)
        X_hat, mean, stddev = model(image.unsqueeze(0))
        #breakpoint()
        image2 = get_image_ready(X_hat*255)
        #breakpoint()
        psnr.append(cv2.PSNR(image1, image2))

    return np.mean(psnr)

def calc_lpips():
    batch, y = next(iter(model.testdataloader))
    lpips_score = []
    for image in batch:
        image1 = get_image_ready(image=image)
        X_hat, mean, stddev = model(image.unsqueeze(0))

        image2 = get_image_ready(X_hat)

        resize_transform = Resize((256, 256))  # Adjust the size as needed

        image1, image2 = torch.tensor(image1), torch.tensor(image2)

        image1 = torch.unsqueeze(image1, 0) 

        image2 = torch.unsqueeze(image2, 0)

        image1 = resize_transform(image1)
        image2 = resize_transform(image2)
        # Initialize the LPIPS metric
        lpips_metric = lpips.LPIPS(net='alex')

        # Calculate LPIPS
        distance = lpips_metric(torch.tensor(image1), torch.tensor(image2))
        lpips_score.append(distance.item())

    return np.mean(lpips_score)

def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)

def get_image_ready(image):
    first_image = np.squeeze(image.detach().numpy())
    # if first_image.shape == (3, 32, 32):
    #     #print('shape detected : 3, 32, 32')
    #     first_image = np.transpose(first_image, (1, 2, 0))

    
    return first_image


def calculate_fid(model, image1, image2):

    with torch.no_grad():
        act1 = model(image1)
        act2 = model(image2)
    breakpoint()
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    mu1, mu2 = mu1.numpy(), mu2.numpy()
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calc_fid_score():
     
    batch, y = next(iter(model.testdataloader))
    fid_scores = []
    real_images = []
    generated_images = []
    images = 0
    n = 5
    while n*n > images:
        for i, image in enumerate(batch):
            image1 = get_image_ready(image=image)
            X_hat, mean, stddev = model((image).unsqueeze(0))

            image2 = get_image_ready(X_hat)
            # img1 = Image.fromarray(np.uint8(image1*255))
            # img2 = Image.fromarray(np.uint8(image2*255))


            # image1 = preprocess(img=img1)
            # #image1 = np.transpose(image1, (1, 2, 0))

            # #image1 = image1.unsqueeze(0)
            # image2 = preprocess(img=img2)
            #image2 = np.transpose(image2, (1, 2, 0))
            
            #image2 = image2.unsqueeze(0)

            real_images.append(image1)
            generated_images.append(image2)
            images += 1
    real_images = scale_images(real_images, (3, 299,299))
    generated_images = scale_images(real_images, (3, 299,299))
    
    breakpoint()

    real_images = torch.tensor(np.array(real_images))
    generated_images = torch.tensor(np.array(generated_images))
    fid_score = calculate_fid(model= inception_model, image1= real_images, image2= generated_images)
    fid_scores.append(fid_score)

    return np.mean(fid_scores)

def generate_metrics():
    psnr = calc_psnr()
    lpips = calc_lpips()
    #fid_score = calc_fid_score()

    print ('The average psnr is : ', psnr)
    print ('The average lpips is : ', lpips)
    #print ('The average fid_score is : ', fid_score)
    
if __name__ == '__main__':
    fid = generate_metrics()
  
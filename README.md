# Unsupervised Ancient Document Image Denoising Based on Attention Mechanism

# Motivation

The digitization of ancient documents is on the rise, while the poor quality of raw manuscripts creates problems for researchers and readers. Thus, we hope to propose a method to reduce noise on images and improve their quality. However, paired ancient document images are almost non-existent. Therefore, we proposed a model that could be trained on unpaired images.

<img src="https://github.com/RylonW/CycleSE-GAN/blob/main/pic/origin_ancient.jpg" width=30% height=30%>

# Dataset
  We chose Fangshan Shijing as our data source and cropped 1200 positive and negative 256 * 256 patches each. The ratio of training set to test set is 5:1. Here are two samples:
  
 <img src="https://github.com/RylonW/CycleSE-GAN/blob/main/pic/samples.jpg" width=20% height=20%>

 Download : [Noise2Denoise](https://www.dropbox.com/scl/fo/jjwrzkzvu0rrtgoaglbz2/h?rlkey=87w58wlp7qwnielm8md2hy9nd&dl=0)
  
|Name| Explaination|
|------|------|
|trainA | clean images for training|
|trainB | noisy images for training|
|testA | clean images for testing|
|testB | noisy images for testing|
|testBgt_sim.txt | character level annotation for testB (simplified chinese)|
|testBgt_tra.txt | character level annotation for testB (traditional chinese)|

The pdf file is the data source from which we have extracted all the samples. You can use it as you like. 
# Network Architecture
<img src="https://github.com/RylonW/CycleSE-GAN/blob/main/pic/generator.jpg" width=50% height=50%>

Our improvements focus on the generator module, which works by embedding the attention module in the stacked residuals module. We hope to focus the feature map on the foreground or background of the image for the purpose of denoising but not changing the text. We have tried two different attention mechanisms: SE and CBAM. The former focuses on the channel dimension only, while the latter focuses on both the channel dimension and the spatial dimension.

  
# Results
<img src="https://github.com/RylonW/CycleSE-GAN/blob/main/pic/denoise_result.png" width=50% height=50%>

## Attention Map Visualization
<img src="https://github.com/RylonW/CycleSE-GAN/blob/main/pic/attention_map.jpg" width=30% height=30%>

## FID metrics
| Feature Dimention  | 64 | 192    | 2048  |
|--------|------------|-------   |--------|
| CycleGAN | 1.34    | 8.95   | 66.47 | 
| CycleGAN+CBAM | 3.16    | 11.52   | 67.18 |
| CycleGAN+SE | **1.06**    | **4.85**   | **59.79** | 

## OCR engine recognition output
We chose Paddle-OCR for testing because it has the best performance on denoised images. Here is the result of a degraded full page after it has been processed by our model.

<img src="https://github.com/RylonW/CycleSE-GAN/blob/main/pic/recognition.png" width=50% height=50%>
## OCR metrics

| model  | CER(%) | 
|--------|------------|
| CycleGAN | 49.66    |
| CycleGAN+CBAM | 36.01    |
| CycleGAN+SE | **31.09**    |

# Configurations
torch 1.9.1


:cherries:More deltials will be added soon!

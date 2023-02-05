# Unsupervised Ancient Document Image Denoising Based on Attention Mechanism

# Requirement
  torch 1.9.1

# Dataset
  We chose Fangshan Shijing as our data source and cropped 1200 positive and negative 256*256 patches each. The ratio of training set to test set is 5:1. Here are some samples:
  
# Network Architecture
<img src="https://github.com/RylonW/CycleSE-GAN/blob/main/pic/generator.jpg" width=50% height=50%>

Our improvements focus on the generator module, which works by embedding the attention module in the stacked residuals module. We hope to focus the feature map on the foreground or background of the image for the purpose of denoising but not changing the text. We have tried two different attention mechanisms: SE and CBAM.

  
# Results
![image](https://github.com/RylonW/CycleSE-GAN/blob/main/pic/denoise_result.png)


## FID metrics
| Feature Dimention  | 64 | 192    | 2048  |
|--------|------------|-------   |--------|
| CycleGAN | 1.34    | 8.95   | 66.47 | 
| CycleGAN+CBAM | 3.16    | 11.52   | 67.18 |
| CycleGAN+SE | 1.06    | 4.85   | 59.79 | 

## OCR engine recognition output
![image](https://github.com/RylonW/CycleSE-GAN/blob/main/pic/recognition.png)
We chose Paddle-OCR for testing because it has the best performance on denoised images.
## OCR metrics
| model  | CER(%) | 
|--------|------------|
| CycleGAN | 49.66    |
| CycleGAN+CBAM | 36.01    |
| CycleGAN+SE | 31.09    |

:cherries:More deltials will be added soon!This is not the final version.

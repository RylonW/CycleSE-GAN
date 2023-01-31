# CycleSE-GAN
Unsupervised Ancient Document Image Denoising Based on Attention Mechanism

# Requirement
  torch 1.9.1

# Dataset
  We proposed a new dataset, which consists of 2400 ancient document images.
  
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
## OCR metrics
| model  | CER(%) | 
|--------|------------|
| CycleGAN | 49.66    |
| CycleGAN+CBAM | 36.01    |
| CycleGAN+SE | 31.09    |

:cherries:More deltials will be added soon!This is not the final version.

VGG-16 implementation using TF Estimator API

Aims
1. Training - Design a runnable training pipeline. Not planning to run at all (dont think I have the bandwidth).
2. Serving - Serve a downloaded vgg16 model
3. Fine tuning or transfer learning - Download pretrained model and fine tune

Also do:
4. Find out various validation errors (top-1, top-5, top-10 etc. for a prediction)
5. Build tf records on a smaller subset of ImageNet (For training)
6. Try training on KSM V-100

paper link - https://arxiv.org/pdf/1409.1556

## ARCHITECTURE
- We do VGG-16, which is 13 Conv layers, followed by 3 FC layers
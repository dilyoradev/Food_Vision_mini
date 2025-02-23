# Food Vision Mini 🍽️  

This project implements a **food classification model** using **EfficientNet-B0** with **pretrained weights** from `torchvision.models.EfficientNet_B0_Weights.DEFAULT`. The model was fine-tuned on a custom food dataset to classify different types of food items.  

## 🏋️ Training Details  

I used **transfer learning** by leveraging a pretrained **EfficientNet-B0** model, replacing its classifier layer, and training it on a custom dataset.  

**Pretrained Weights:**  
```python
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
```
## Custom Image Inference
After training, I tested the model on custom food images, and the predictions were:

Custom Image 1: Predicted Probability → 0.512
Custom Image 2: Predicted Probability → 0.503
This suggests that the model is somewhat confident in its predictions, though further fine-tuning could improve results.

## Technologies Used
PyTorch for model training and evaluation
`EfficientNet-B0` (Pretrained Weights) for feature extraction
Transfer Learning to fine-tune the classifier layer
Custom Dataset for improved food classification

## Future Improvements
Expand dataset for better generalization
Further fine-tune EfficientNet-B0's feature extractor layers
Experiment with data augmentation and hyperparameter tuning

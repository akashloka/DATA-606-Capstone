## CAPSTONE PROJECT REPORT

## Deciphering Earth’s Surface – Land Cover Prediction via Computer Vision on Satellite Imagery

### Submitted by
Akash Reddy – LB04019
Yashwanth Vemulapalli – PP24297

### 1. Title and Author
* Project Title : Deciphering Earth’s Surface – Land Cover Prediction via Computer Vision on Satellite Imagery
* Prepared for UMBC Data Science Master Degree Capstone by Dr. Tony Diana
* Course Code: Data 606
* Presentation file: 

 
### 2. ABSTRACT

Nowadays, effective continuous environmental monitoring, disaster response, and urban planning are vital for the development that satisfies the current needs while safeguarding the capacity of future generations to meet their own. In this project, we aimed to address the critical need for an automated image recognition system capable of predicting land cover in satellite imagery. This system would facilitate ongoing observation of shifts in the environment, prompt reactions to disasters, and well-informed urban development strategies.  

To build this Image recognition system we used EuroSAT Sentinel 2 satellite images which had 27000 images belonging to 10 different classes. Each image in the dataset had 13 spectral bands. We experimented with 4 different architectures [Custom CNN model, Pretrained VGG19+ Pretrained VGG16, Pretrained Xception] and could achieve a maximum accuracy of 97.13% on unseen test data using VGG19. Further, the web application created using the Streamlit was deployed using NGROK. 

### 3.	INTRODUCTION

In this technological era, technology can be leveraged in environmental science to give us amazing chances to understand and manage our planet resources. One such area is use of satellite image data for environmental surveillance wherein careful watch is kept for tracking changes, assessing impacts, and ensuring well-being of ecosystems and communities. This information collected through environmental surveillance serves to identify potential risks, guide decision-making processes, and support initiatives aimed at preserving and safeguarding natural resources.

The environmental surveillance can be achieved by training and deploying the CNN architecture model. 

### 4.	DATASET
The dataset was created by the Remote Sensing Technology institute of the German Aerospace Centre. The EuroSAT dataset contains high resolution Sentinel-2 satellite images from the European Space Agency’s Copernicus program. There are total of 27000 images distributed among 10 classes. Each image in the dataset had 13 spectral bands. The images cover various landscapes such as urban areas, farmland, forest, sea and lake, residential areas, annual crops, bare land and more. The size of the dataset is 2.7 GB. 

Link to download - https://madm.dfki.de/files/sentinel/EuroSATallBands.zip

### 5.	LITERATURE REVIEW

This literature review explores key research papers focusing on different applications of CNNs in remote sensing domain.

Helber et al. (2017) [1]: 

Helber et al. tackle the task of classifying land use and land cover using Sentinel-2 satellite imagery, which is readily available through the Copernicus Earth observation initiative. They present a new dataset containing Sentinel-2 images across 13 spectral bands and encompassing 10 land cover categories, comprising a total of 27,000 annotated images. Employing advanced deep convolutional neural networks (CNNs), they achieve an impressive overall accuracy of 98.57% in classification. This study underscores the potential of CNN-based classification systems in Earth observation applications, particularly in identifying changes in land use and land cover patterns over time.

Senecal et al. (2019) [2]: 

Senecal et al. concentrate on crafting neural network designs customized for multi-spectral and hyper-spectral images, areas that have not garnered as much focus as RGB imagery. They present a compact CNN architecture tailored to effectively classify 10-band multi-spectral images, showcasing better classification accuracy and efficacy in utilizing samples compared to conventional deep architectures like ResNet or DenseNet.

Shaha and Pawar (2018) [3]: 

Shaha and Pawar delve into the utilization of CNNs, specifically leveraging pre-trained models such as VGG19, for tasks related to image classification. They apply transfer learning to refine pre-existing networks on recognized datasets like GHIM10K and CalTech256, evaluating the efficacy of VGG19 against other well-known CNN structures like AlexNet and VGG16. Furthermore, they explore hybrid learning strategies that combine CNN feature extraction with support vector machine (SVM) classifiers. Their results illustrate the efficacy of fine-tuning VGG19 in achieving superior performance in image classification endeavors, underscoring the significance of CNN architectures and transfer learning methodologies.

### 6.	EXPLORATORY DATA ANALYSIS

<img width="470" alt="image" src="https://github.com/akashloka/DATA-606-Capstone/assets/82594243/58fa4c12-e4ce-414b-b2e9-7b1ea5b81a47">

 
Fig.1 – Bar Plot for Distribution of land cover class labels
From the above plot it can be observed that the Pasture image label has least number of images in dataset and whereas the below mentioned labels has the highest number of images.
•	Herbaceous Vegetation
•	Forest
•	Sealake
•	Annual Crop
•	Residential
 
Fig.2  – Line chart showing channel size variation
From the above line chart, it can be observed that there are same number of channels for all the images in the dataset which are 13.
 
Fig.3– Sample images from the dataset.
 
Fig.4 – Line chart for heights and widths of images
From the above line charts for image heights and widths, it can be observed that the heights and widths of all images are found to be 64 and 64 respectively.

### 7.	METHODOLOGY 
For building this image recognition system, the following methodology has been employed.

•	Download the dataset from the source.
•	Explore the image data.
•	Create a custom data generator and process .tiff image files using tifffile library.
•	Create a CNN model and use pre-trained CNN models (VGG16, VGG19, Xception).
•	Train the model.
•	Deploy the web app created using Streamlit in NGROK.

### 8.	RESULTS
The following are the training results obtained for each model for the following conditions.

Optimizer – Adam @ initial learning rate of 0.0001
Total number of epochs – 60.
Early stopping – Training stops early if the categorical accuracy does not change for 4 consecutive iterations.
Learning rate – Reduces learning rate by 0.95 if the categorical accuracy does not change for 2 consecutive iterations.

Model	Params	Train Loss	Test Loss	Train Accuracy	Test Accuracy
Custom CNN model	114394	0.1567	0.1438	0. 946	0. 9515
VGG19	20551852	0. 1086	0. 1051	0. 9647	0.9713
VGG16	15242156	0. 1067	0. 0961	0. 9651	0. 9689
Xception	22907284	0. 1122	0. 2156	0. 9614	0.9607
Table.1 – Summary of the Models
 
Fig.5 – Loss and Accuracy plot for CNN Model
 
Fig.6 – Loss and Accuracy plot for 	VGG19 Model
 
Fig.7 – Loss and Accuracy plot for VGG16 Model
 
Fig.8 – Loss and Accuracy plot for Xception Model

### 9.	CONCLUSION

The Custom CNN model, with 114,394 parameters, achieved respectable results with a train accuracy of 94.6% and a test accuracy of 95.15%. However, both VGG19 and VGG16 outperformed the Custom CNN model in terms of accuracy. VGG19, with 20,551,852 parameters, demonstrated the highest test accuracy of 97.13%, closely followed by VGG16 with a test accuracy of 96.89%. VGG models also exhibited lower test loss values compared to the Custom CNN model.

On the other hand, the Xception model, despite having the highest number of parameters (22,907,284), showed slightly lower performance compared to VGG19 and VGG16, with a test accuracy of 96.07% and a relatively higher test loss.
Therefore, for land cover image classification, it can be concluded that VGG19 and VGG16 are the most effective models among those tested, offering superior accuracy and lower loss values.

### 10.	REFERENCES

1.	Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217-2226.
Link - https://ieeexplore.ieee.org/abstract/document/8736785
2.	Senecal, J. J., Sheppard, J. W., & Shaw, J. A. (2019, July). Efficient convolutional neural networks for multi-spectral image classification. In 2019 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.
Link - https://ieeexplore.ieee.org/abstract/document/8851840
3.	Shaha, M., & Pawar, M. (2018, March). Transfer learning for image classification. In 2018 second international conference on electronics, communication and aerospace technology (ICECA) (pp. 656-660). IEEE.
Link - https://ieeexplore.ieee.org/document/8474802

# Final Report

**Author**: Zhenzhao Xu zhenzhao@upenn.edu, Xiong Zheng zhxiong@upenn.edu 

# Introduction

In this project, we try to use convolutional neural network to train the model and detect the outlines and areas of buildings. After the training, we will apply the model to developing cities in the dataset to understand the trends (heated areas) of urban development. The use case of this model is to advice the real estate investment.

# Data

| Data                      | Source                             | Description                                                  |
| ------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| SpaceNet Satellite Images | https://spacenet.ai/sn7-challenge/ | This dataset consists of Planet satellite imagery mosaics, which includes 24 images (one per month) covering ~100 unique geographies. The dataset will comprise over 40,000 square kilometers of imagery and exhaustive polygon labels of building footprints in the imagery, totaling over 10 million individual annotations. |

Imagery consists of RBGA (red, green, blue, alpha) 8-bit electro-optical (EO) monthly mosaics from Planet’s Dove constellation at 4 meter resolution. For each of the Areas Of Interest (AOIs), the data cube extends for roughly two years, though it varies somewhat between AOIs. All images in a data cube are the same shape, though some data cubes have shape 1024 x 1024 pixels, while others have a shape of 1024 x 1023 pixels. Each image accordingly has an extent of roughly 18 square kilometers.

<center><a href="https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development">Spacenet 7 Multi-Temporal Urban Development | Kaggle</a><br>The SpaceNet has already labeled the outlined footprint of each building.</center>

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/sn7_gif.gif" width=50%>

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/image-20220429223103611.png" width=50%>

​	There are 1408 items in the data. Within each item,there are a high-resolution images (1024*1024 RGB) and a GeoJson file with geometry attributes of buildings. Some images have the shape of 1023 * 1023 instead of 1024 * 1024. So to format the data, we enlarge all images into 1024 * 1024 pixels by filling black pixels to the right and bottom edges of those images. We don't have to do the same change to GeoJson, with its coordinates starting at the upper left point of the image. So it fits well with those modified satellite images.

```python
def imagePathToArray(path):
  # read satellite img as numpy array
  image = Image.open(path)
  image = np.array(image)[:,:,:3]
  # fill black pixels to the right and bottom edges of 1023*1023 images
  zeros = np.zeros((1024,1024,3), dtype=np.uint8)
  zeros[:image.shape[0],:image.shape[1],:] = image
  return zeros
```

​	And to make the training process efficient, we split each `Satellite Image` and `Building Masks` (1024 * 1024) to 16 images (256 * 256).

```python
def splitArray(arr, splitNum=4):
  # split the image into smaller chunks
  return [np.hsplit(x, splitNum) for x in np.vsplit(arr,splitNum)]
```

# Methods

Based on the use case, we choose U-Net as the model for image segmentation, the concept is shown as follows:

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/YUKcLpTCB67fXTURoo6dmCGPIa2gMN2GtesUdf2kFuglWHQ3Wi5_UrB8Do9v16qzQCKYL0c6WTfGON1hzK5fFmbr2rrfH7liuW9j4DM_0bDBC9gfR9mPYUc9r1xgkRDdmCVVLbD3eT5_" width=80%>

*Reference: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)*

# Results

​	In this project, we try to use convolutional neural network to train the model and detect the outlines and areas of buildings. The use case of this
model is to advice the real estate investment. After using the UNet structure, our Method One has achieved convincing results of detecting building outlines. So we stick to the Method One and try to tune the parameter to achieve a better learning and predicting effect, instead of settle for using the alternative Method Two . The parameters we finally used in the model are listed as follows.

```python
epochs = 35
batch_size=10
optimizer="rmsprop"
loss="sparse_categorical_crossentropy"
```

​	The training process is shown as follows. SHOULD BE REPLACED. We can see from the graph that, as the number of epochs increases, the accuracy of the model steadily increases and the loss steadily decreases. However, because of the limitation of the computing power, we stick to the  35 epochs.

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/image-20220429221108020.png" width=70%>

​	The final effect is shown below. We can see that model really does a good great job beyond our expectations, not only distinguishing the building outlines clearly but also telling the difference between rural land and urban area. The final testing accuracy for the testing dataset is 95.92%.

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/rural.png" width=70%>

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/urban.png" width=70%>

​	After obtaining a satisfying model, we apply the model to our use case that is to detect the urban development movement under the temporal context. Therefore, we choose 25 month satellite images of Asunción, the capital city of Paraguay. The city experienced a fast development in the past. And we apply the model on the images to see the urban development between January 2018 to January 2021.  In the following three month (Jan.2018, Jan.2019, Jan 2020), results are displayed. One thing needs to be noted is that the images below is the concatenation of the model output. (The resolution of below images is 1028*1028, the resolution of the output of the model is 256\*256).

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/2018_01.jpg" width=70%>

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/2019_01.jpg" width=70%>

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/2020_01.jpg" width=70%>

​	We can see from the images that Asunción has increased its density and broadened its built boundaries into the green lands during these 25 months. And also in the image, the west side has experienced a tenser development.

# Discussion


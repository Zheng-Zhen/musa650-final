# Progress Report

**Author**: Xiong Zheng zhxiong@upenn.edu , Zhenzhao Xu zhenzhao@upenn.edu 



## Progress

### Download Data

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/image-20220416145044040.png" width=60%>

It takes effort to download the image data from the original SpaceNet S3 bucket. Instead, we obtain the same data from [Kaggle](https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development), there are 1408 items in the data. Within each item,there are a high-resolution images (1024*1024 RGB) and a GeoJson file with geometry attributes of buildings. 

### Format Data with missing pixels

Some images have the shape of 1023 * 1023 instead of 1024 * 1024. So to format the data, we enlarge all images into 1024 * 1024 pixels by filling black pixels to the right and bottom edges of those images. We don't have to do the same change to GeoJson, with its coordinates starting at the upper left point of the image. So it fits well with those modified satellite images.

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

### Rasterize GeoJson Data

```python
buildingMask = features.rasterize(geom, out_shape=(1024,1024))
```
We are using `rasterio` to rasterize GeoJson Data of buildings as `Building Masks`. The masks are taken as the output of the Image Segmentation model. The shape of mask is the same as the satellite image (1024 * 1024 * 1)

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/image-20220416150309266.png" width=60%>

### Split Images

To make the training process efficient, we split each `Satellite Image` and `Building Masks` (1024 * 1024) to 16 images (256 * 256).

```python
def splitArray(arr, splitNum=4):
  # split the image into smaller chunks
  return [np.hsplit(x, splitNum) for x in np.vsplit(arr,splitNum)]
```



## Next Steps & Challenges

- Based on the use case, we choose U-Net as the model for image segmentation, the concept is shown as follows:

<img src="https://raw.githubusercontent.com/ShaunZhxiong/ImgGarage/main/ShaunZhxiong/ImgGarage/img/image-20220416152209812.png" width=60%>

*Reference: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)*



- However, the bandwidth of google Colab is not stable so it is practically impossible (4 hours) to load the 22,529 images data and masks. (Both in `.png` format.) So we have to convert it into `.npy` (Numpy array binary storage files) file in our local computers and import it into Colab as a whole. 
- Importing all 22,529 data at once at the beginning of the modeling process may also be a challenge to our RAMs and Neural Network Training. A better solution will be reading a batch of data at a time. However, to achieve that, over-complicated tensorflow hand-written codes are required.
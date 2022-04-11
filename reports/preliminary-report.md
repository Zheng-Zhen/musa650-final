# Preliminary Report

**Author**: Xiong Zheng zhxiong@upenn.edu , Zhenzhao Xu zhenzhao@upenn.edu

- [Preliminary Report](#preliminary-report)
  - [Abstract](#abstract)
  - [Project Method](#project-method)
    - [Method 0](#method-0)
    - [Method 1](#method-1)
  - [Data sources](#data-sources)

## Abstract

In this project, we try to use convolutional neural network to train the model and detect the outlines and areas of buildings. After the training, we will apply the model to developing cities in the dataset to understand the trends (heated areas) of urban development. The use case of this model is to advice the real estate investment.

## Project Method

### Method 1

```mermaid
graph TB
0[Satellite Images]
1[Unet Segmentation Model]
2[Satellite Building Masks]
3["Mask Comparing Algorithm (24 Masks Monthly in a developing city)"]
4[Heatmaps of Building Footprint Changes]
0 --> 1 --> |"is building or not?"|2 --> 3 --> 4
```

### Method 2

```mermaid
graph TB
0[Statelite Images]
1[CNN to Count the Building Number]
2[Compare the building counts across years]
0 --> 1 --> 2
```

## Data sources

| Data                      | Source                             | Description                                                                                                                                                                                                                                                                                                                    |
| ------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SpaceNet Satellite Images | https://spacenet.ai/sn7-challenge/ | This dataset consists of Planet satellite imagery mosaics, which includes 24 images (one per month) covering ~100 unique geographies. The dataset will comprise over 40,000 square kilometers of imagery and exhaustive polygon labels of building footprints in the imagery, totaling over 10 million individual annotations. |

Imagery consists of RBGA (red, green, blue, alpha) 8-bit electro-optical (EO) monthly mosaics from Planet’s Dove constellation at 4 meter resolution. For each of the Areas Of Interest (AOIs), the data cube extends for roughly two years, though it varies somewhat between AOIs. All images in a data cube are the same shape, though some data cubes have shape 1024 x 1024 pixels, while others have a shape of 1024 x 1023 pixels. Each image accordingly has an extent of roughly 18 square kilometers.

*[Spacenet 7 Multi-Temporal Urban Development | Kaggle](https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development)*

![](https://spacenet.ai/wp-content/uploads/2020/06/sn7_gif.gif)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4101651%2F88990ba121d3b550820b72caeebdbef6%2Flabels.png?generation=1605457001725966&alt=media)
*The SpaceNet has already labeled the outlined footprint of each building.*
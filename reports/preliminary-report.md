# Preliminary Report

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
1[Unet Segmentation Model]
2[Satellite Building Masks]
3["Mask Comparing Algorithm (24 Masks Monthly in a developing city)"]
4[Heatmaps of Building Footprint Changes]
1 --> |"is building or not?"|2 --> 3 --> 4
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

| Data                      | Source                             | Description                                                  |
| ------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| SpaceNet Satellite Images | https://spacenet.ai/sn7-challenge/ | This dataset consists of Planet satellite imagery mosaics, which includes 24 images (one per month) covering ~100 unique geographies. The dataset will comprise over 40,000 square kilometers of imagery and exhaustive polygon labels of building footprints in the imagery, totaling over 10 million individual annotations. |

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4101651%2F88990ba121d3b550820b72caeebdbef6%2Flabels.png?generation=1605457001725966&alt=media)
*The SpaceNet has already labeled the outlined footprint of each building.*
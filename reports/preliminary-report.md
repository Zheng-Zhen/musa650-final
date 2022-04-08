# Preliminary Report

- [Preliminary Report](#preliminary-report)
  - [Abstract](#abstract)
  - [Project Method](#project-method)
    - [Method 0](#method-0)
    - [Method 1](#method-1)
  - [Data sources](#data-sources)

## Abstract

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4101651%2F88990ba121d3b550820b72caeebdbef6%2Flabels.png?generation=1605457001725966&alt=media)
*The SpaceNet labeling team  painstakingly outlined the footprint of each building.*

## Project Method

### Method 0

```mermaid
graph TB
1[Unet Segmentation Model]
2[Satellite Building Masks]
3["Mask Comparing Algorithm (24 Masks Monthly)"]
4[Heatmaps of Building Footprint Changes]
1 --> |"is building or not?"|2 --> 3 --> 4
```

### Method 1

```mermaid
graph TB
1[Unet Segmentation Model]
2[Satellite Building Masks]
3["Mask Comparing Algorithm (24 Masks Monthly)"]
4[Heatmaps of Different Types of Building Footprint Changes]
1 --> |"What type of building it is?"|2 --> 3 --> 4
```

## Data sources

| Data                      | Source                             | Description                                                                                                                                                                                                                                                                                                                    |
| ------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SpaceNet Satellite Images | https://spacenet.ai/sn7-challenge/ | This dataset consists of Planet satellite imagery mosaics, which includes 24 images (one per month) covering ~100 unique geographies. The dataset will comprise over 40,000 square kilometers of imagery and exhaustive polygon labels of building footprints in the imagery, totaling over 10 million individual annotations. |
|                           |                                    |                                                                                                                                                                                                                                                                                                                                |
|                           |                                    |                                                                                                                                                                                                                                                                                                                                |

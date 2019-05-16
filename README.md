Identifying human-made water reservoirs using satellite imagery

Group members: Achyut Dave and Kylen Solvik

Introduction

In the Brazilian agricultural frontier, cattle ranchers have dammed streams to create tens of thousands of small water impoundments as  ear-round water sources for their livestock. These impoundments have important implications for stream health and greenhouse gas emissions. Automatically identifying small cattle reservoirs and distinguishing them from other water bodies remains challenging. The recent growth of aquaculture in the region has made the task even more difficult. Farmers and ranchers dig small ponds on their land to raise farmed fish. These are particularly challenging to distinguish from cattle ranching reservoirs. Some previous studies have explored reservoir classification. However, most have focused on larger reservoirs, such as those created by hydroelectric dams.

Identifying small reservoirs (as small as 100 m 2 in our case) created using earthen dams is a significant added challenge. Previous research has relied primarily on temporal and/or spectral data. For example, Khandelwal et al. developed a MODIS-based surface water classification algorithm using spectral features. They achieved high accuracy and created a product available at 8-day temporal resolution, but at 500 m spatial resolution. Even with a 0.5 water fraction threshold applied to each pixel, the smallest detectable reservoir was 62,500 m 2 , 2+ orders of magnitude above what we require. Additionally, this method did not incorporate water body shape, which could help performance. Reservoirs tend to be triangular: broader near the dam and tapering to a thin stream at the top. This distinctive shape can be used to identify anthropogenic impoundments.

While temporal approaches can provide important year-to-year context, non-temporal methods have an advantage in terms of data availability. Time-based approaches require multi-year data to work, while a purely spectral- and/or shape-based method can run on a single year of data. This also allows us to use new satellites with higher spatial and spectral resolution, such as the European Space Agency’s Sentinel satellites which have only been operational for the last couple years. Landsat (which is commonly used for tasks like this) has a very long operational history, with data available back to the 1980s. However, it only has 30m resolution, which is too large to capture many of the smaller reservoirs. Further, the Landsat data changes between satellites in the series. Landsat 5 provides data back to 1984, but has a much worse signal-to-noise ratio than later instruments. The Scan Line Corrector on the Landsat 7 instrument failed in 2003 (4 years after launch), creating a “striping” effect in that data where strips of each scene are missing. Landsat 8 is a highly precise instrument, benefiting from decades of development, but only came online in 2013. Using a time series from different Landsat satellites requires correcting for these differences. In comparison, the ESA Sentinel-2 satellite provides similar multispectral data to Landsat 8, but with a lower return interval and higher resolution (mostly 10m and 20m, depending on the band). Non-temporal approaches allow us to use this high quality data rather than being hamstrung by Landsat’s issues.

We approach the task of reservoir identification using using semantic segmentation, which assigns meaningful classes to regions in an image based on training data. This makes for an interesting data structure. The training images themselves do not have classes assigned to them. Instead, each pixel within each image has a class assigned to it. This structure created some problems when trying to implement boosting this semester.

The “U-Net” fully convolutional neural network architecture was first developed for biomedical imaging semantic segmentation applications, but has been successfully applied to satellite remote sensing data. The architecture (Figure 1) makes use of a 10 layer deep neural network. Each convolutional layer is followed by the Rectified Linear United (“ReLU”) activation function. This activation function was first developed with a lot of mathematical justifications. It is the most popular activation function in deep neural networks and is more effective with respect to its counterparts, such as Logistic Sigmoid, or hyperbolic tangent functions. An extra layer of “max-pooling” operations is added after the activation. The basic functionality of max-pooling operation is to downsample the size of data for the proper processing in the next layer. It passes only the maximum value among the neighbors to the next convolutional layer, thus representing the values of the whole neighborhood. The network is designed to capture both local information and large-scale context, making it well-suited to remote sensing classification tasks.

A previous Kaggle competition required users to perform pixel-wise landcover classification on high-resolution satellite images. Semantic segmentation approaches based on U-Net generally produces the best performance. This Kaggle competition provided a good resource for building our own classifier Also, researchers in Earth Lab at CU are using U-Net classifiers for urban surface classification (concrete vs. natural) and wildfire location prediction. The architecture is clearly well-suited for spatial and/or
remote sensing tasks. Compared to a traditional sliding window approach, the U-net performs faster and is more precise during image segmentation problems.

Previously, Kylen and others worked on building a classifier using Sentinel-2 satellite imagery (10m resolution red, green, blue, and near-infrared bands) with a U-Net. The goal was to utilize both shape and spectral attributes to identify human-made impoundments. The U-Net classifies each pixel in the training images as reservoir (the positive class) or non-reservoir (the negative class). While overall accuracy of this original classifier was relatively high (>95%), it performed worse in certain landscapes. In particular, there were many false positives in wetlands and floodplains. Further, overall accuracy is not a good metric. The dataset is highly imbalanced: non-reservoir pixels vastly outnumber reservoirs.

At the beginning of the semester, we proposed improving the performance of the classifier by labeling more training examples, tuning the existing U-Net CNN, and testing other architectures and methods. We also proposed adding more satellite imagery: Sentinel-1 radar data, more Sentinel-2 spectral bands, and historical Landsat (30m resolution) data.

We explored many of these options and were able to improve the performance of the classifier by 17% (evaluated using F1 score on a development set). Below, we discuss our work this semester. First we’ll cover the baseline classifier that we started with. Then we will outline the modifications we explored as well as future directions that could yield more improvements.

Baseline Method

As mentioned, Kylen and others had already developed a basic classifier for small-reservoir classification. In this section we will outline this classifier, which we use as a baseline for further modifications.

First, Sentinel-2 scenes from the dry season of the target year (2017) were identified using Google Earth Engine. These scene were then filtered for cloudiness, discarding any scenes with a cloud cover percentage over 10% (cloud cover fraction is provided in the scene metadata). The remaining scenes were mosaiced by taking the median value of the scenes for each pixel and spectral band. The result was a continuous, consistent satellite image for the entire region of interest. Figure 2 shows the study region with the Sentinel-2 RGB mosaic created for this study. The satellite data was exported to Google Cloud Storage and random 500x500 images were
extracted from the mosaic for training and testing.

Human annotators (including Kylen) created training data from these 500x500 images. We used LabelBox ( https://labelbox.com ) to outline all regions in each image that were cattle ponds. The process is slow, as it requires carefully zooming into areas of interest to determine whether a blob of bright pixels represents a cattle reservoir or if it is only a natural pond, a shadow, or something else. Reservoir segmentation is challenging even for human eyes. In total, 493 images (each 500x500 pixels) were labeled in this way. 143 images were skipped because they were too challenging to label by hand. The skipped images might create bias in our accuracy estimates
upwards. Ideally, annotators would indiscriminately label images in random order, but there is no feasible alternative to skipping when the image cannot be annotated. After labeling, the outlined polygons are converted to binary masks with 1 values representing a cattle reservoir and 0 for everything else.

The Sentinel-2 satellites provides multi-band spectral data primarily at 10 and 20m resolution, with two bands at 60m. Four bands are available at 10m resolution: red, green, blue, and near-infrared (NIR). The original approach used only these 10m bands, plus two band ratios useful for vegetation and water detection. The Normalized Difference Vegetation Index (NDVI) is derived from the NIR and red bands, while the Normalized Difference Water Index (NDWI) is based on the NIR and green bands. In total, 6 image bands were used for the original classification.

After the labeling and satellite imagery extraction, we had both a binary mask representing the reservoir locations and a 6-band image containing the satellite data for each of the 493 images in our dataset. These images were split into training, development, and test sets using a 70/15/15 split. The training set was then augmented by applying a random rotation and magnification to each image, doubling the effective training set.

In the baseline, we did not perform significant tuning on the U-Net. It was trained using a dice coefficient loss function, a learning-rate of 1E-5, the “Adam” optimizer, and early-stopping when the validation dice coefficient did not improve for 8 consecutive epochs.

Methods: Improving the Classifier

We explored a number of options for improving the reservoir classifier: adding more satellite bands, using a different loss function, implementing boosting via AdaBoost, and tweaking the learning rate. Some were more successful than others. We also added new evaluation metrics.

Evaluation

Previously, we evaluated the classifier performance using overall accuracy, true positive rate, false positive rate, and the dice coefficient for the validation/development set. As discussed above, overall accuracy is not very helpful. Because the classes are
highly imbalanced, it is possible to achieve high accuracy by assigning all pixels to the 0 (non-reservoir) class. True and false positive rates are useful, but do not paint the full picture.

We added functions to calculate precision, recall, and F1 score in Keras. Then we used the final validation scores to evaluate whether our modifications were resulting in classification improvement.

For the baseline classifier, the best F1 score was 0.58. This was the number we aimed to improve this semester. Satellite Bands We tested 10 more radar and spectral bands as inputs to the classifier. Of these, 6 were Sentinel-2 20m spectral bands, 2 were Sentinel-1 10m radar bands, and 2 were new spectral ratio bands like NDWI and NDVI. Because labeled training examples had already been created using 500x500 pixel subsets of the original Sentinel-2 10m mosaic, we had to match the new bands exactly to these existing training images. This proved to be non-trivial, and took a couple weeks to finish. It would have been much more straightforward if Kylen had written the original code to include geographic projection and transformation information for all the randomly sampled subset images, so it was karma that he had to deal with this problem. First, we had to calculate the corner coordinates for the existing training images. We used the spatial information from the original mosaics and calculated the offset for each training subset based on pixel size and subset location within the mosaics.

Given the corner coordinates, you can easily extract matching images from the new mosaics using the geospatial data abstraction library (GDAL). However, this only works when the pixel size matches between the old and new images. The 20m bands did not always align with the 10m pixels. We first had to resample the data to 10m using nearest neighbor resampling via gdalwarp. It would be more efficient to keep the 20m bands at their native resolution and adapt the U-Net architecture to fit, but resampling them to match the other bands’ resolution was more simple.

With the addition of two shortwave infrared (SWIR) bands, we could calculate additional band ratios. There are two variants of NDWI that we thought could be useful:

Modified NDWI (MNDWI) and Gao 1996’s NDWI. Both were designed to detect water content in vegetation, rather than surface water. They utilize mid-infrared (MIR) spectra, which roughly match the Sentinel-2 SWIR bands that were added. The band ratios for each are shown below in comparison with the original NDWI we used in the baseline.

NDWI = Green + NIR
Green − NIR
MNDWI = Green + MIR
Green − MIR
NDWIGao = NIR + MIR
NIR − MIR

Loss Functions

We tested 4 loss functions: dice coefficient loss, Jaccard loss, binary cross-entropy, and weighted variant of the dice coefficient.
The weighted dice coefficient loss was intended to correct for the imbalanced training data. Negative class (non-reservoir) pixels are much more frequent than positive (reservoir) pixels. We explored several ways to correct for this. The most common is to use undersampling from the majority class to balance the data, but there is no simple way to apply this to our data because each training example in our dataset is composed of both the negative and positive class. Another method is to use a weighted loss function that places more weight on the minority class, forcing the algorithm to pay more attention to it. We doubled the weight of positive class pixels in our dice coefficient loss function.

Learning Rate and Early Stopping

Learning rate is an important parameter for training a neural network. Too high and the training may blow up or never converge to a reasonable solution. Too low and training will take longer and may settle on a local loss minimum instead of finding the global optimum. In some crude previous tests, learning rates between 1E-5 and 1E-4 were found to be effective. This semester, we explored a broader range of learning rate settings and performed many training runs to tune this parameter.

In some tests, we noticed that performance varied significantly between very similar parameter settings. We hypothesized that the network was not always converging, or was converging to a local rather than global minimum in the loss function. To address this, we dramatically increased the patience in the early-stopping during training. Originally, training stopped when the network saw no improvements in the validation loss for 8 epochs. In our final tests, we increased this to 20 epochs to be sure that the network was reaching convergence before quitting. After this, we saw more consistent results. In most cases, the network reached the early-stopping criterion within 100 epochs.

Boosting

Boosting is a machine learning ensemble meta-algorithm primarily used in supervised learning problems. Boosting was developed to improve the performance of ensembles of weak classifiers / learners.

In layman terms, boosting helps in classification problems by building the model from the training data, then creating a second model that attempts to correct the errors from the first model. Therefore, while predicting the final output, boosting keeps track of the errors of the first model. Most popular boosting methods include AdaBoost and Stochastic Gradient Boosting.

AdaBoost algorithm was developed for binary classification in 1999. The weights of weak classifiers are combined to form a weighted sum that represent the final output.

Each classifier is re-trained and more accurate classifier is assigned higher weight to increase its impact on the final outcome. AdaBoost is popular due its ability to reduce overfitting, especially for imbalanced datasets.

AdaBoost could not be directly used for our dataset, which is a multi-dimensional dataset. But we looked into the methods through which it can be implemented and found out that there are two methods. First, we could convert all our multi-dimensional data into a long 1-dimensional array and apply the AdaBoost. Second, we would have to use AdaBoost Classifier multiple times to the different bands of images, and treat them as different 2D images. In both cases there is a chance of loss of very important spatial data, and thus we would require some time to work on how can the AdaBoost be implemented perfectly on our multi-dimensional dataset.

Training

The NN was trained on a Google Cloud Compute Engine with a GPU. The network was implemented in keras with a tensorflow backend. Using 16 bands increased memory usage. We changed from an Nvidia Tesla K80 GPU to a P100 on Google Cloud Platform. As a bonus, the faster training times allowed us to iterate more quickly, experimenting with different band combinations, learning rates, and loss functions. We were also able to increase the batch size from 8 to 12. The more powerful GPU does cost more. A K80 costs just $0.45 per hour, while the P100 costs $1.46, more than triple. The higher price was partially offset by the faster training times.

Because of the additional satellite bands, we also had to increase the memory of the base virtual machine just to load the training and test images into memory.

Previously, we were able to use an n1-highmem-4 instance type, with 26 GB of RAM. We had to upgrade to a n1-highmem-8 with 52 GB, adding another $0.23 per hour. The instance also comes with an extra 4 CPU cores, but because training only utilized the GPU these sat idle. It would help to find a way to integrate the CPU into NN training.

Alternatively, we could load the images in batches so that we could use a machine with less memory, although the added read/write operations might impact performance. Despite these extra costs, we only used a total of ~$200 in Google Cloud credits this semester. That included all data preparation and storage as well as training and tuning our model, although >50% of it was just the P100 GPU hours. Overall, this is relatively affordable for the high performance resources were used.

Grid Search to Optimize Hyperparameters

We performed a gridsearch over a total of 4 different loss functions, 4 band combinations, and 5 learning rates to find the optimal hyperparameters. We compared performance using F1 score on the development set.

Results

Two loss functions (binary cross-entropy and Jaccard coefficient loss) did not converge. Regardless of learning rate, these loss functions caused the network to predict all negative pixels.

Our best performance (F1 = 0.68) was achieved using the original 6-bands from the baseline (red, green, blue, near-IR, NDVI, and NDWI). The weighted dice loss generally outperformed the unweighted version. The difference between the 14-band and 16-band classifiers show that adding the new normalized-difference band ratios (MNDWI and NDWI Gao ) boosted the accuracy.

Band combinations were as follows: 6-bands = red, green, blue, near-infrared, NDWI, and NDVI. 10-bands: 6b + 2x radar, MNDWI, and NDWI Gao . 14-bands: 6b + 2x radar, 3x red-edge, narrow NIR, and 2x SWIR. 16-bands: 14b + MNDWI and NDWI Gao .In this case, the major reservoirs are all identified, but there are areas of false positives. These are likely small natural ponds, which can be hard to distinguish from human-made impoundments.

Some may also be reservoirs that the human annotator missed due to small size.

Discussion and Future Directions

We achieved significant performance increase over our baseline classifier. Careful learning-rate tuning using a grid search seems to be responsible for a large part of this. Other contributions came from dramatically increasing the patience of the early-stopping mechanism (from 8 to 20 epochs) and weighting our loss function to correct the imbalanced classes.

Adding 10 more image band options did not improve the classifier’s F1 score, which was very surprising. There could be a few reasons for this. First, we could test more band combinations. Perhaps an 8-band classifier with the original 6 bands, MNDWI, and NDWI Gao would perform best. Second, many of the new bands were only available at 20m resolution and had to be resampled to 10m. They may not provide the fine-scale information needed for the classifier. Finally, the U-Net architecture does not convolve between multiple bands, instead relying on 2D convolutions and combining the per-band outputs in the final layers. The original paper on biomedical image segmentation only used single-band images. The four band ratios that we calculated (NDWI, NDWI Gao , MNDWI, NDVI) help address this lack of integration, but this could be extended much further. An easy option would be to calculate more band ratios. A more robust solution would be to modify the network architecture to integrate multiple bands via 3D convolutions. In fact, there have been some papers applying this approach to biomedical image segmentation. This could be a fruitful avenue for future work.

Originally, we proposed adding temporal information. We did not prioritize this because there are advantages to only using the best data from a single year (as discussed in the Introduction). Single-year data maximizes the flexibility of the method, but it is possible we are sacrificing some accuracy. There are a couple ways we could try integrating temporal information. First, we could add historical Landsat bands to the network. For example, we could extracting matching spectral images from 2000 using Landsat 7. Including these bands might allow the network to identify regions where water has appeared in the last 15-20 years as cattle ranching has expanded in the
region and more dams have been built. Another way of incorporating temporal data would be to look at the seasonality of the water bodies. Perhaps reservoir spectral and shape attributes vary more (or less) between the seasons that natural water bodies. We could add satellite data from the wet season to the U-Net to capture this. However, it is very challenging to extract decent wet season images from our study region. During the wet season, it is cloudy more often than not. Given these challenges, it is hard to say whether we would be able to improve performance by adding temporal data, but it is certainly a path to explore.

We suspect that the single biggest improvement we could make would be to label more training data. While we could spend more time tweaking and tuning the classifier and adding more feature, we may be capped by the amount of training examples. 493 images is not a lot of data for a complex task like this. Image augmentation helped us create more training data, but that also has limits. Adding an extra 100-200 images, particularly if they are concentrated in difficult landscapes like river floodplains, could boost the performance. 

Over break, we could try to annotate more examples. The ultimate goal of this project is to provide reservoir count, size, and location
information that can be combined with field data on methane emissions from reservoirs to more fully understand the impacts of these water impoundments. The PI on this project (Marcia Macedo) has been gathering these field measurements over the past two seasons. We hope to publish our results within the next year or two. Our current reservoir classification may be good enough to provide approximate numbers, but it would be great if we could raise the F1 score a little higher. We could also publish amethods-focused paper on the classification. As mentioned in the introduction, previous approaches have been at a much coarser resolution and have not incorporated
reservoir shape into their classification process.

Conclusion

This semester, we explored and tested a variety of ways to improve the performance of our reservoir classification process. To start, we added precision, recall, and F1 score calculations to help evaluate our modifications. Then, we added more satellite bands, tested a range of loss functions, and performed a grid search to optimize classification performance. We found that a weighted dice coefficient loss function with a higher learning rate (7.5E-5) and the original 6 image bands provided the best performance. We achieved a final F1 score of 0.68, roughly a 17% improvement over our baseline method from the start of this semester.

There are more options for improving on our results, although the most valuable might be to just label more training data. Through our work this semester, we learned about fine-tuning a neural network to optimize performance. This experience can be applied to many future data science tasks.

References

[1] A. Khandelwal, A. Karpatne, M. E. Marlier, J. Kim, D. P. Lettenmaier, and V. Kumar, “An approach for global monitoring of surface water extent variations in reservoirs using MODIS data,” Remote Sensing of Environment, vol. 202, pp. 113–128, Dec. 2017.
[2] B. L. Markham, J. C. Storey, D. L. Williams, and J. R. Irons, “Landsat sensor performance: history and current status,” IEEE Transactions on Geoscience and Remote Sensing, vol. 42, no. 12, pp. 2691–2694, Dec. 2004.
[3] S. K. McFEETERS, “The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features,” International Journal of Remote Sensing, vol. 17, no. 7, pp. 1425–1432, May 1996.
[4] B. Gao, “NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space,” Remote Sensing of Environment, vol. 58, no. 3, pp. 257–266, Dec. 1996.
[5]  Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016, 2016, pp. 424–432.

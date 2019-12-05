# LandCover_ML
* James Sullivan
* jesull2@bu.edu
* Updated 12/5/2019

**************************************************************************************************
This project has produced Convolutional Neural Nets (CNNs) and Support Vector Machines (SVMs)
that classify aerial or satellite imagery by predominant land-cover type, and/or development status.

The binary and categorical CNNs trained for this project generally take 4-band imagery as input,
and output the name of a land-cover class, as defined by the NOAA Office for Coastal Management’s 
Regional Land Cover Classification Scheme. 

Two CNN models created for this project utilized the model structure and weights of a previously
built/trained CNN (VGG16), a process called transfer learning. 

Raw aerial photographs and land-cover imagery, identified respectively as
“2016 NAIP 4-Band 8 Bit Imagery of Massachusetts” and “2016 High Res Land Cover of Massachusetts,”
was obtained from NOAA.  

Training and test datasets were created by cutting the raw images into smaller tiles
(using Geospatial Data Abstraction Library; GDAL), and extracting pixel values as numpy arrays. 

**************************************************************************************************

To predict land-cover classification and determine whether or not land has been developed, please 
run Predict_Images.py from the Testing module. 

Convolutional Neural Net models are defined in keras_models.py from the Training module.
CNNs are trained using RunTraining.py.
SVMs are trained using SVM_Training.py

To create your own dataset, use region_data_prep.py in the DataPreparation module.

**************************************************************************************************

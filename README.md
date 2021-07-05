
---
# Geospatial Non-Local Mean Filter (geoNLMF)

[Saif Aati](mailto:saif@caltech.edu) :  saif@caltech.edu

[Saif Aati](mailto:saifaati@gmail.com) :  saifaati@gmail.com

---

## **Overview**
This tool is a modified implementation of the Non-Local Means algorithm used for image denosing.
This algorithm has demonstrated an ability to preserve fine details while reducing additive white Gaussian noise.
The implementation provided here extends the method to filter and denoise geospatial displacement maps derived using 
image correlation technique.
The proposed package is a modified version of the algorithm proposed in [[1]](#1) and
an open-source version of the implementation in[[2]](#2).

## **Installation**
To install geoNLMF from PyPI:

    pip install geoNLMF

Building from source:

    git clone...
    cd geoNLMF
    pip install -e .


### **Dependencies**

#### *Core requirements*
*Python 3* and C++>11. 

The package is tested on 3.6+ and C++11. 

[geoRoutines](https://github.com/SaifAati/geoRoutines), [GDAL](http://gdal.org),
[numpy](http://www.scipy.org) , [scipy](http://numpy.org), [matplotlib](http://matplotlib.org).

## **Examples**:

![Example1](https://github.com/SaifAati/geoNLMF/blob/main/Test/Data/ResExample1.png)

---
## References:
<a id="1">[1]</a> A. Buades, B. Coll and J. -. Morel, "A non-local algorithm for image denoising," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), 2005, pp. 60-65 vol. 2, doi: 10.1109/CVPR.2005.38.

<a id="1">[2]</a> S. Leprince, S. Barbot, F. Ayoub and J. Avouac, "Automatic and Precise Orthorectification, Coregistration, and Subpixel Correlation of Satellite Images, Application to Ground Deformation Measurements," in IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 6, pp. 1529-1558, June 2007, doi: 10.1109/TGRS.2006.888937.

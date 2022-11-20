//
// Created by Saif Aati on 5/27/21.
// saif@caltech.edu | saifaati@gmail.com
//
#include <omp.h> // OpenMP threading : sudo apt-get install libomp-dev
#include <xmmintrin.h>//Need this for SSE intrinsics
#include <emmintrin.h>//Need this for SSE2 intrinsics.
#include <limits>//For quiet_nan() function
#include <float.h>//For "_isnan" function.

#include <bits/stdc++.h>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
#include <array>

#include "geoNLMF.h"
#include "geoNLMFLibTypeCode.h"

extern "C"
{

    void InputData(
                    int *debug,
                    int *patchSize,
                   int *searchAreaSize,
                   float *h,
                   int *weightingMethod,
                   float *minWeightValue
                   ,int *minNumberOfWeights,
                   int *linear,
                   int *adaptive,
                   int*SNRint,
                   int *CenterInclusive,
                   int *dimension,
                   float *iArray_fl,
                   float *oArray_fl,
                   float *snrArray)
    {
        CNLMFilter filterObject;

       
        

        if (*debug ==1)
        {
             std::cout<< "=== geoNLMF(C++) ===="<<std::endl;
        }
        int bandNumber = 1;  // @SA: is this variable useful !!

        //Patch size:
        filterObject.SetWindowSize((int) *patchSize);

        //Search size:
        filterObject.SetSearchSize((int) *searchAreaSize);

        //H noise estimation parameter:
        filterObject.StoreHparam((double) *h);

        //Weighting method:
        filterObject.StoreWeightingFunctionType((int) *weightingMethod);

        //Minimum weight threshold:
        filterObject.StoreMinimumWeightValue((float) *minWeightValue);

        //Mimimum number of weights:
        filterObject.StoreMinNumberOfWeights((int) *minNumberOfWeights);

        //Store the linear fit option (1 is user wants linear regression, 0 for simple averaging):
        // Linear fi option:
        //  1: For linear regression (Future Work)
        //  2: For gaussian Kernel (Future work)
        //  0: Average Kernel
        filterObject.StoreLinearFitOption((int) *linear);

        //Store the adaptive H option (true = 1, false = 0):
        filterObject.StoreAdaptiveHoption((int) *adaptive);

        //Store whether or not the user wants to use SNR weighting:
        filterObject.StoreSNRWeightingOption((int) *SNRint);

        //Store the center point option (1 = include, 0 = omit):
        filterObject.StoreCenterPointOption((int) *CenterInclusive);

        //Data height and width dimensions:

        int dataDimensions[2] = {*dimension, *(dimension+1)};
        filterObject.SetDataHeight((int) dataDimensions[0]);
        filterObject.SetDataWidth((int) dataDimensions[1]);

        filterObject.ComputeOutputDimensions();

        //Allocate the NaNMask array:
        filterObject.AllocateNaNMask();

        // Set the input data type float:
        filterObject.iDataType = geoNLMFLib::FLOAT;

        //Store the pointer to the input array:
        filterObject.StoreInputArrayPtr(iArray_fl);

        //Store the pointer to the output array:
        int MemoryByteAlignment = 32;

        float *oArray = (float *) _mm_malloc(*dimension * *(dimension+1) * bandNumber * 4, MemoryByteAlignment);
        filterObject.StoreOutputArrayPtr(oArray);

//        //Compute gaussian window:
//        filterObject.ComputeGaussianWindow_Float();

        if (filterObject.UsesSNRdata())
        {
            filterObject.StoreSNRArrayPtr(snrArray);
            //Allocate SNR mask array:
            filterObject.AllocateSNRMask();
        }
        // Get the number of CPUs and store for use by OpenMP:
        //SYSTEM_INFO sysInfo;
        //GetSystemInfo(&sysInfo);
        //filterObject.NumCPUs = sysInfo.dwNumberOfProcessors;
        filterObject.nbCPUs = omp_get_num_procs();

        // Reset thread sync variable:

        filterObject.synchronizationCounter = 0;

        // Check the filterData type and generate the NaN mask:
        if (filterObject.iDataType == geoNLMFLib::FLOAT)
        {
        #pragma omp parallel num_threads(filterObject.nbCPUs)
            {
                filterObject.GenerateNaNMask();
            }
        }

        // Reset the synchronization counter:
        filterObject.synchronizationCounter = 0;
        // Check to see if the SNR weighting option has been chosen:
        if (filterObject.UsesSNRdata())
        {
//            std::cout<<"Using SNR Filtering!!" <<std::endl;
            // user wants SNR weighting applied. Check the filterData
            // type and replace the NaNs in the filterData with 0 while,
            // at the same time, building an SNR-NaN mask for use
            // after the filtering process is complete (to restore
            // the replaced NaN values so that we don't permanently
            // alter the user's SNR filterData):

            if (filterObject.iDataType == geoNLMFLib::FLOAT) {
                #pragma omp parallel num_threads(filterObject.nbCPUs)
                {
                    filterObject.ReplaceNaNValsInSNRArray();
                }

                filterObject.synchronizationCounter = 0;

                #pragma omp parallel num_threads(filterObject.nbCPUs)
                {
                    filterObject.Filter_SNR();
                }

                filterObject.synchronizationCounter = 0;

                #pragma omp parallel num_threads(filterObject.nbCPUs)
                {
                    filterObject.RestoreNaNValsInSNRArray();
                }
            }

            filterObject.synchronizationCounter = 0;

        }
        else
        {
//            std::cout<<"Not using SNR Filtering!!" <<std::endl;
            if (filterObject.iDataType == geoNLMFLib::FLOAT)
            {
            #pragma omp parallel num_threads(filterObject.nbCPUs)
            {
                filterObject.Filter();
           }

                filterObject.synchronizationCounter = 0;

            #pragma omp parallel num_threads(filterObject.nbCPUs)
                {
                    filterObject.PlaceNaNValsInOutputArray();
                }
            }
        } // end of "else not using SNR option" block.

        __attribute__((aligned(16))) int edgeWidth = filterObject.windowSize / 2;

        //Finally, copy the unfiltered edges of the noisy input data
        //into their respective positions in the output array:

        if (filterObject.iDataType == geoNLMFLib::FLOAT)
        {
            int dataWidth = filterObject.GetDataWidth();
            int dataHeight = filterObject.GetDataHeight();
            int edgeOffset = (filterObject.GetWindowSize() / 2);//Use int truncation.
            int oHeight = filterObject.GetOutputHeight();
            int oWidth = filterObject.GetOutputWidth();

            float *noisyData = filterObject.GetInputArrayPtr();
            float *filteredData = filterObject.GetOutputArrayPtr();

            unsigned char *nanMask = filterObject.GetNaNMask();

            unsigned int topEdgePosition;
            unsigned int bottomEdgePosition;
            unsigned int leftEdgePosition;
            unsigned int rightEdgePosition;

            for (int x = 0; x < edgeOffset; x++) {
                for (int y = 0; y < dataWidth; y++) {
                    //copy values from input array to output array
                    //for top and bottom edges.

                    //Top edge:
                    topEdgePosition = (dataWidth * x) + y;

                    //Check NaNMask for presence of NaN:
                    if (*(nanMask + topEdgePosition) == 0) {
                        //NaN present. Assign to location:
                        *(filteredData + topEdgePosition) = std::numeric_limits<float>::quiet_NaN();

                    } else {
                        *(filteredData + topEdgePosition) = *(noisyData + topEdgePosition);

                    }

                    //Bottom edge:
                    bottomEdgePosition = ((oHeight + edgeOffset + x) * dataWidth) + y;

                    //Check for presence of NaN:
                    if (*(nanMask + bottomEdgePosition) == 0) {
                        //NaN present. Assign to location:
                        *(filteredData + bottomEdgePosition) = std::numeric_limits<float>::quiet_NaN();

                    } else {
                        *(filteredData + bottomEdgePosition) = *(noisyData + bottomEdgePosition);

                    }
                }
            }
            //Now restore the left and right edges:
            for (int x = 0; x < dataHeight; x++) {
                for (int y = 0; y < edgeOffset; y++) {
                    //copy values from input array to output array
                    //for left and right edges.

                    //Left edge:
                    leftEdgePosition = (dataWidth * x) + y;

                    //Check NaNMask for presence of NaN:
                    if (*(nanMask + leftEdgePosition) == 0) {
                        //NaN present. Assign to location:
                        *(filteredData + leftEdgePosition) = std::numeric_limits<float>::quiet_NaN();

                    } else {
                        *(filteredData + leftEdgePosition) = *(noisyData + leftEdgePosition);

                    }

                    //Right edge:
                    rightEdgePosition = (dataWidth * (x + 1)) - edgeOffset + y;

                    //Check for presence of NaN:
                    if (*(nanMask + rightEdgePosition) == 0) {
                        //NaN present. Assign to location:
                        *(filteredData + rightEdgePosition) = std::numeric_limits<float>::quiet_NaN();

                    } else {
                        *(filteredData + rightEdgePosition) = *(noisyData + rightEdgePosition);

                    }
                }
            }
            for (int i=0; i< (dataHeight*dataWidth);i++)
            {
                *(oArray_fl+i) = *(filteredData+i);
            }
        }

    }

}

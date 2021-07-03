
#include <float.h>//For "_isnan" function.
#include <limits>//For quiet_nan() function
#include <cmath>
#include <cstdio>
#include <omp.h> // OpenMP threading : sudo apt-get install libomp-dev

#include <xmmintrin.h>//Need this for SSE intrinsics
#include <emmintrin.h>//Need this for SSE2 intrinsics.

#include "Include/geoNLMF.h"

#include <iostream>
// Constructors:
CNLMFilter::CNLMFilter() {}

void CNLMFilter::SetSearchSize(int size) {
    searchSize = size;
}

void CNLMFilter::SetWindowSize(int size) {
    windowSize = size;
}

int CNLMFilter::GetWindowSize() {
    return windowSize;
}

void CNLMFilter::StoreWeightingFunctionType(int type) {
    weightingFunctionType = type;
}

void CNLMFilter::StoreMinNumberOfWeights(int number) {
    minNumberOfWeights = number;
}

void CNLMFilter::StoreMinimumWeightValue(double value) {
    minimumWeightThreshold = value;
}

double CNLMFilter::GetMinimumWeightThreshold() {
    return minimumWeightThreshold;
}

void CNLMFilter::StoreLinearFitOption(int selection) {
    useLinearFit = selection;
}

void CNLMFilter::StoreAdaptiveHoption(int selection) {
    useAdaptiveH = selection;
}

void CNLMFilter::StoreSNRWeightingOption(int selection) {
    useSNRdata = selection;
}

bool CNLMFilter::UsesSNRdata() {
    bool returnBool;
    if (useSNRdata == 1) { returnBool = true; }
    else { returnBool = false; }

    return returnBool;
}

void CNLMFilter::StoreCenterPointOption(int selection) {
    //Store the center point option (1 = include, 0 = omit):
    centerPointInclusive = selection;
}

void CNLMFilter::StoreHparam(double h) {
    hParam = h;
    std::cout<<"Noise param:"<<hParam<<std::endl;
}

void CNLMFilter::SetDataWidth(int width) {
    dataWidth = width;
}

void CNLMFilter::SetDataHeight(int height) {
    dataHeight = height;
}

int CNLMFilter::GetDataHeight() {
    return dataHeight;
}

int CNLMFilter::GetDataWidth() {
    return dataWidth;
}

void CNLMFilter::ComputeOutputDimensions() {
    //compute the dimensions of the output array minus the unfiltered edges
    int edgeWidth = windowSize / 2; //Use int truncation.
    oHeight = dataHeight - (2 * edgeWidth);
    oWidth = dataWidth - (2 * edgeWidth);
    std::cout<<"oDim:"<< oHeight<<" "<<oWidth<<std::endl;
}

void CNLMFilter::AllocateNaNMask() {
    NaNMask = (unsigned char *) _mm_malloc((dataWidth * dataHeight), 16);
}

void CNLMFilter::StoreInputArrayPtr(float *inputPtr)
{
    iData = inputPtr;
}
void CNLMFilter::StoreOutputArrayPtr(float *outputPtr) {

    oData = outputPtr;
}

void CNLMFilter::ComputeGaussianWindow() {

    float sigma = (((float) windowSize) - 1.0f) / 2.0f;
    float **gaussianKernel;
    gaussianKernel = new float *[windowSize];//Rows.
    for (int row = 0; row < windowSize; row++) {
        gaussianKernel[row] = new float[windowSize];//Columns.
    }

    int windowCenter = windowSize / 2;

    for (int row = 0; row < windowSize; row++) {
        for (int column = 0; column < windowSize; column++) {
            //Compute distance from center of kernel window:
            float rowDistance = (float) (windowCenter - row);
            float columnDistance = (float) (windowCenter - column);
            float distance = sqrt((rowDistance * rowDistance) + (columnDistance * columnDistance));

            gaussianKernel[row][column] = expf(-0.5f * ((distance / sigma) * (distance / sigma)));
        }
    }

    //Check the center point included option.
    // If excluded,set the inner gaussian value to zero:
    if (centerPointInclusive == 0) {
        int center = windowSize / 2;//Use integer truncation.
        gaussianKernel[center][center] = 0.0f;
    }

    //Allocate the 1D memory array, aligned on 16-byte boundary. This consists of the gaussian coefficients plus the
    // zero buffering needed for the vectorization.
    // Each vector consists of 4 floats, so to compute the buffer size, divide the total number of window elements
    //by 4 and round up to the next integer:
    float nbBufferedElts = (float) (windowSize * windowSize);
    int nbVectors = (int) (ceil(nbBufferedElts / 4.0f));

    gaussianWindow = (float *) _mm_malloc((nbVectors * 4 * 4), 16);

    //Loop through and load buffer with zeros:
    int pos = 0;
    for (int x = 0; x < (nbVectors * 4); x++) {
        *(gaussianWindow + pos) = 0.0f;
        pos++;
    }

    //Loop through and load the gaussian values:
    pos = 0;
    for (int row = 0; row < windowSize; row++)//"column" is correct.
    {
        for (int column = 0; column < windowSize; column++)//"row" is correct.
        {
            *(gaussianWindow + pos) = gaussianKernel[row][column];
            pos++;
        }
    }

    //Deallocate temporary GaussianKernel memory:
    for (int x = 0; x < windowSize; x++) {
        delete[] gaussianKernel[x];
    }
    delete[] gaussianKernel;
}

int CNLMFilter::GetOutputHeight() {
    return oHeight;
}

int CNLMFilter::GetOutputWidth() {
    return oWidth;
}

float *CNLMFilter::GetInputArrayPtr() {
    return iData;
}

float *CNLMFilter::GetOutputArrayPtr() {
    return oData;
}

void CNLMFilter::StoreSNRArrayPtr(float *snrPtr) {

    SNR = snrPtr;
}

void CNLMFilter::AllocateSNRMask() {
    SNRMask = (unsigned char *) _mm_malloc((dataWidth * dataHeight), 16);

}

void CNLMFilter::GetSearchLimits(int XPos, int YPos, int searchDimension, int dataWidth,
                                 int dataHeight, int *left, int *top) {
    //We need to determine the physical extents of the search area
    //i.e. the coords in the data matrix (actually only the left and top) based
    //on the current position in the data matrix and the edge dimension
    //of the user-specified search area. There are 9 possibilities.
    //Use a series of "if-else" statements to determine which one we
    //are dealing with and set the search limits.

    int searchEdgeDim = searchDimension / 2;//Use int truncation.

    if ((XPos <= searchEdgeDim) && (YPos <= searchEdgeDim))//Possibility 1
    {
        //The upper left corner up to and including the first
        //position wherein the search area will increment along
        //with the X and Y pos.
        *(left) = 0;
        *(top) = 0;
    } else if (((XPos > searchEdgeDim) && (XPos < (dataWidth - searchEdgeDim - 1))) &&
               (YPos <= searchEdgeDim))//Possibility 2
    {
        //The upper middle portion wherein the search area slides
        //horizontally, but not vertically, with the filtering window:
        *(left) = (XPos - searchEdgeDim);
        *(top) = 0;
    } else if ((XPos >= (dataWidth - searchEdgeDim - 1)) && (YPos <= searchEdgeDim))//Possibility 3
    {
        //The upper right corner where the search window cannot slide
        //with the filtering window:
        *(left) = (dataWidth - searchEdgeDim * 2 - 1);
        *(top) = 0;
    } else if ((XPos <= searchEdgeDim) && ((YPos > searchEdgeDim) &&
                                           (YPos < (dataHeight - searchEdgeDim - 1))))//Possibility 4.
    {
        //The middle left portion where the search window increments vertically
        //but not horizontally with the sliding filter window:
        *(left) = 0;
        *(top) = (YPos - searchEdgeDim);
    } else if (((XPos > searchEdgeDim) && (XPos < (dataWidth - searchEdgeDim - 1))) &&
               ((YPos > searchEdgeDim) && (YPos < (dataHeight - searchEdgeDim - 1))))//Possibility 5.
    {
        //The middle of the data array, where the search window shifts with the filtering
        //window.
        *(left) = (XPos - searchEdgeDim);
        *(top) = (YPos - searchEdgeDim);
    } else if ((XPos >= (dataWidth - searchEdgeDim - 1)) &&
               ((YPos > searchEdgeDim) && (YPos < (dataHeight - searchEdgeDim - 1))))//Possibility 6.
    {
        //The middle right-hand portion of the array, where the search area can increment
        //vertically but not horizontally:
        *(left) = (dataWidth - searchEdgeDim * 2 - 1);
        *(top) = (YPos - searchEdgeDim);
    } else if ((XPos <= searchEdgeDim) && (YPos >= (dataHeight - searchEdgeDim - 1)))//Possibility 7
    {
        //The lower left hand portion of the array, where the search area cannot increment
        //with the filtering window:
        *(left) = 0;
        *(top) = (dataHeight - searchEdgeDim * 2 - 1);
    } else if (((XPos > searchEdgeDim) && (XPos < (dataWidth - searchEdgeDim - 1))) &&
               (YPos >= (dataHeight - searchEdgeDim - 1)))//Possibility 8
    {
        //The bottom middle portion of the data array, where the search area can increment
        //horizontally with the filter window but not vertically:
        *(left) = (XPos - searchEdgeDim);
        *(top) = (dataHeight - searchEdgeDim * 2 - 1);
    } else//Possibility 9
    {
        //The lower right portion of the data array, where the search area cannot
        //increment with the filtering window:
        *(left) = (dataWidth - searchEdgeDim * 2 - 1);
        *(top) = (dataHeight - searchEdgeDim * 2 - 1);
    }
}

void CNLMFilter::GenerateNaNMask() {
    //Acquire a scanline for processing:
    int row;
    #pragma omp critical
    {
        row = synchronizationCounter; // Acquire counter value.
        synchronizationCounter++; // Increment counter.
    }

    while (row < dataHeight) {
        for (int column = 0; column < dataWidth; column++) {
            //Get the filterData value and check for NaN condition:
            if (__isnan(iData[(row * dataWidth) + column])) {
                //Value is NaN. Set the value in the mask to 0:
                NaNMask[(row * dataWidth) + column] = 0;

                //Replace the unfiltered NaN with zero:
                iData[(row * dataWidth) + column] = 0.0f;
            } else {
                NaNMask[(row * dataWidth) + column] = 1;
            }
        }

        //Acquire another scanline for processing:
    #pragma omp critical
        {
            row = synchronizationCounter; // Acquire counter value.
            synchronizationCounter++; // Increment counter.
        }
    }
}

void CNLMFilter::ReplaceNaNValsInSNRArray() {
    //Grab a scanline for processing:
    int row;
#pragma omp critical
    {
        row = synchronizationCounter;
        synchronizationCounter++;
    }

    while (row < dataHeight) {
        for (int column = 0; column < dataWidth; column++) {
            //Get the filterData value and check for NaN condition:
            if (__isnan(SNR[(row * dataWidth) + column])) {
                //Value is NaN. Replace the value with 0.0f:
                SNR[(row * dataWidth) + column] = 0.0f;

                //Set the mask with 0 to indicate presence of a NaN:
                SNRMask[(row * dataWidth) + column] = 0;
            } else {
                //Set the value in the mask to 1:
                SNRMask[(row * dataWidth) + column] = 1;
            }
        }

        //Grab the next scanline for processing:
#pragma omp critical
        {
            row = synchronizationCounter;
            synchronizationCounter++;
        }
    }
}

void CNLMFilter::FloatFilter() {

    //Allocate two 1D array of 32-bit floats to hold the filtering window and
    //gaussian values. Buffer this array so that it is a multiple of 4 for SSE:
    int nbWindowElts = windowSize * windowSize;
    __attribute__((aligned(16))) int bufferedNbVectors = (int) ceil(((float) (windowSize * windowSize)) / 4.0f);

    float *slidingWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4), 16);//Sliding window.
    float *NaNMaskWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4), 16);//Sliding window NaN mask.
    float *bckWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4),
                                                   16);//Background values in scanline.
    float *gaussianWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4), 16);//Gaussian window.
    float *NaNNormalizedGaussian = (float *) _mm_malloc((bufferedNbVectors * 4 * 4),
                                                        16);//Gaussian normalized for SNR & NaN "holes".
    float *NaNMaskBackground = (float *) _mm_malloc((bufferedNbVectors * 4 * 4),
                                                    16);//NaN mask for background window.
    float *register_ = (float *) _mm_malloc((4 * 4), 16);//For summing across register_.

    //Load the gaussian values from the filterData object to this local array:
    float *gaussianPtr = gaussianWindow;

    for (int x = 0; x < (bufferedNbVectors * 4); x++)
    {
        *(gaussianWindow + x) = *(gaussianPtr + x);
    }

    //Loop through and load the buffers with zeros:
    for (int x = 0; x < (bufferedNbVectors * 4); x++)
    {
        *(slidingWindow + x) = 0.0f;
        *(NaNMaskWindow + x) = 0.0f;
        *(bckWindow + x) = 0.0f;
        *(NaNMaskBackground + x) = 0.0f;
        *(NaNNormalizedGaussian + x) = 0.0f;
    }

    //Declare variables to store the running totals used in the final step of the filtering algorithm:
    __m128 iterationsSum;
    __attribute__((aligned(16))) float temp;
    __attribute__((aligned(16))) float weightValueSum = 0.0f;
    __attribute__((aligned(16))) float weightValue = 0.0f;
    __attribute__((aligned(16))) float scalarValue3Sum = 0.0f;

    //Precompute a coefficient you'll need during the filtering operation:
    __attribute__((aligned(16))) float hOrig = (float) hParam;
    float negOne = -1.0f;
    float two = 2.0f;
    float zero = 0.0f;
    float one_ = 1.0f;
    float hVal = (float) hParam;

    __m128 hOriginal = _mm_load_ss(&hOrig);
    __m128 h = _mm_load_ss(&hVal);
    __m128 hSquared = _mm_mul_ss(h, h);
    __m128 coefficient = _mm_div_ss(_mm_load_ss(&negOne), _mm_mul_ss(hSquared, _mm_load_ss(&two)));

    //Declare some vectors to be used with the scalar SIMD instructions:
    __m128 std = _mm_load_ss(&zero);
    __m128 n = _mm_load_ss(&zero);
    __m128 mean = _mm_load_ss(&zero);
    __m128 one = _mm_load_ss(&one_);
    //Declare variables used to synchronize threads and control the loops:
    __attribute__((aligned(16))) int currentScanline;
    __attribute__((aligned(16))) int totalScanlines = oHeight;
    __attribute__((aligned(16))) int nbHorizontalShifts = oWidth;
    __attribute__((aligned(16))) int edgeWidth = windowSize / 2;//Use integer truncation;
    __attribute__((aligned(16))) int searchEdgeWidth = searchSize / 2;//Use integer truncation.

    //Compute the number of vertical and horizontal shifts the filtering window
    //makes within the search area:
    __attribute__((aligned(16))) int nbHSearchShifts = searchSize - (edgeWidth * 2);
    __attribute__((aligned(16))) int nbVSearchShifts = nbHSearchShifts;

    //Declare variables for the search area limits:
    __attribute__((aligned(16))) int left, top;


    //Some SSE filterData type vectors we'll need:
    __m128 filterVec, backgroundVec, gaussianVec, result1, result2,
            nanWindowVec, gaussianSum;

    __attribute__((aligned(16))) float originalNoisyValue;

    //Store the weighting type for use later:
    __attribute__((aligned(16))) int weightingType = weightingFunctionType;

    //Store the linear fit bool:
    bool usingLinearFitOption = false;
    if (useLinearFit > 0)
    {
        usingLinearFitOption = true;
        std::cout<<"Using Linear Fit Option: "<<usingLinearFitOption<<std::endl;

    }

    //__declspec(align(16)) float MinimumWeightThreshold = (float)MinimumWeightThreshold;
    __attribute__((aligned(16))) int weightNumberThreshold = minNumberOfWeights;

    //Center of Gaussian array:
    __attribute__((aligned(16))) int gaussianCenterOffset = ((windowSize * windowSize) - 1) / 2;

    //Grab a scanline for processing:
    #pragma omp critical
    {
        currentScanline = synchronizationCounter;
        synchronizationCounter++;
    }

    //unsigned char* NaNMask = NaNMask;
    float *unfilteredData = iData;
    float *filteredData = oData;


    //Loop over filterData and filter:
    while (currentScanline < totalScanlines)
    {
        //Set the coordinate of the center of the filtering window
        //based on the scanline. A scanline of value 0 starts at
        //the row equal to the edge width in the original filterData array:
        int row = currentScanline;

        //Loop over the scanline, extracting the filter windows
        //and sliding them across the background search area, filtering:

        for (int col = 0; col < nbHorizontalShifts; col++)
        {
            //Check to see if the current value is NaN.
            //If so..bypass:
            unsigned char NaNMaskValue = *(NaNMask + ((row + edgeWidth) * dataWidth) + (col + edgeWidth));

            if (NaNMaskValue > 0)
            {
                //Extract the filtering window and it's associated NaN mask
                // and SNR window based on the row and column centers.
                int increment = 0;
                for (int row = 0; row < windowSize; row++)
                {
                    for (int column = 0; column < windowSize; column++) {
                        *(slidingWindow + increment) = *(unfilteredData + ((row + row) * dataWidth) +
                                                         (col + column));
                        *(NaNMaskWindow + increment) = (float) (*(NaNMask + ((row + row) * dataWidth) +
                                                                  (col + column)));
                        increment++;
                    }
                }
                //Check to see if user has selected "adaptive H" option:
                if (useAdaptiveH > 0)
                {
                    //Reset the adaptive variables:
                    //N                 = 0.0f;
                    //Mean              = 0.0f;
                    //std = 0.0f;
                    _mm_store_ss(&zero, n);
                    _mm_store_ss(&zero, mean);
                    _mm_store_ss(&zero, std);

                    //Compute the total number of non-NaN values in the current
                    //filtering window by summing over the NaN mask (in the mask, NaN
                    //values are equal to 0, cast to float in NaNMaskWindow):
                    for (int x = 0; x < (windowSize * windowSize); x++) {
                        if (*(NaNMaskWindow + x) > 0.0f) {
                            n = _mm_add_ss(n, one);
                            mean = _mm_add_ss(mean, _mm_load_ss(slidingWindow + x));
                        }
                    }

                    //Compute window mean:
                    //Mean = Mean / (N*N);
                    mean = _mm_div_ss(mean, _mm_mul_ss(n, n));

                    //Loop through and compute the standard deviation,
                    //again checking for NaNs using the NaN mask:
                    for (int x = 0; x < (windowSize * windowSize); x++) {
                        if (*(NaNMaskWindow + x) > 0.0f) {
                            //float value = *(slidingWindow + x);
                            //std += (Mean - value)*(Mean - value);
                            __m128 value = _mm_load_ss(slidingWindow + x);
                            std = _mm_add_ss(std,_mm_mul_ss(_mm_sub_ss(mean, value), _mm_sub_ss(mean, value)));
                        }
                    }

                    //Final StdDev value for this sliding window:
                    //std = std * (1.0f / (N*N - 1.0f));
                    std = _mm_mul_ss(std, _mm_div_ss(one, _mm_sub_ss(_mm_mul_ss(n,n), one)));

                    //Recompute h, hSquared, and coefficient:
                    //h           = hOriginal * (sqrt(std));
                    //hSquared    = h * h;
                    //coefficient = -1.0f / (2.0f * (hSquared));
                    h = _mm_mul_ss(hOriginal, _mm_sqrt_ss(std));
                    hSquared = _mm_mul_ss(h, h);
                    coefficient =
                            _mm_div_ss(_mm_load_ss(&negOne), _mm_mul_ss(_mm_load_ss(&two), hSquared));

                }//End of "if UsingAdaptiveH" block.

                //Reset the running sum variables to zero:
                weightValueSum = 0.0f;
                scalarValue3Sum = 0.0f;

                //Get the search area limits based on the current filtering pixel.
                //(In hindsight I only needed the left and top values):
                GetSearchLimits(col + edgeWidth, row + edgeWidth, searchSize, dataWidth,
                                dataHeight, &left, &top);

                //Initialize the weight counter and threshold:
                int nbWeightsAboveThreshold = 0;

                //Loop over search area, sliding window and filtering:
                for (int vPos = 0; vPos < nbVSearchShifts; vPos++)
                {
                    for (int hPos = 0; hPos <  nbHSearchShifts; hPos++)
                    {

                        //Check for the presence of a NaN at the center of the
                        //background search patch. Bypass filtering operation
                        //if NaN is present:
                        float centerValue =
                                (float) (*(NaNMask + ((top + vPos + edgeWidth) * dataWidth) +
                                           (left + hPos + edgeWidth)));



                        if (centerValue > 0.0f)
                        {
//                            std::cout<<"CenterValue:"<<CenterValue<<" ";
                            //Extract background window and NaN mask into array (NaNMaskWindow
                            //was extracted earlier):
                            increment = 0;
                            for (int row = 0; row < windowSize; row++)
                            {
                                for (int column = 0; column < windowSize; column++) {
                                    *(bckWindow + increment) =
                                            *(unfilteredData + ((top + vPos + row) * dataWidth) +
                                              (left + hPos + column));

                                    *(NaNMaskBackground + increment) =
                                            (float) (*(NaNMask + ((top + vPos + row) * dataWidth) +
                                                       (left + hPos + column)));

                                    increment++;
                                }
                            }

                            //Normalize the Gaussian kernel to the sum of all the gaussian
                            //values in the kernel that do not correspond to a NaN value in
                            //either the background NaN mask or the window NaN mask.
                            //Use SSE (exclude SSE3 summing across register_ instruction):
                            float gaussianNormalizationTerm = 0.0f;
                            for (int x = 0; x < nbWindowElts; x += 4)
                            {
                                //Combine NaN masks with bitwise AND operation. Just use the
                                //nanWindowVec variable:
                                nanWindowVec =
                                        _mm_and_ps(_mm_load_ps(NaNMaskWindow + x), _mm_load_ps(NaNMaskBackground + x));

                                //Multiply the Gaussian by the NaN mask to filter with NaN:
                                gaussianVec =
                                        _mm_mul_ps(_mm_load_ps(gaussianWindow + x), nanWindowVec);

                                //Store this result to the gaussian buffer for use later, and
                                //extract and sum (no SSE3) to normalize:
                                _mm_store_ps(register_, gaussianVec);//for accumulation
                                _mm_store_ps((NaNNormalizedGaussian + x), gaussianVec);//For final normalization.

                                //Accumulate:
                                gaussianNormalizationTerm +=
                                        (*(register_) + *(register_ + 1) + *(register_ + 2) + *(register_ + 3));
                            }

                            //Now we have to check to make sure that the gaussian array doesn't consist
                            //of zeros. This is only a concern if the user has opted to exclude the
                            //central value during the processing operation, so check for
                            //that condition first:
                            if (centerPointInclusive < 1)
                            {
                                //They are excluding the center point, so we have to
                                //check the gaussian and make sure all the elements
                                //sum up to more than zero:
                                int increment = 0;
                                float accumulatedGaussian = 0.0f;
                                for (int a = 0; a < windowSize; a++) {
                                    for (int b = 0; b < windowSize; b++) {
                                        accumulatedGaussian += *(NaNNormalizedGaussian + increment);
                                        increment++;
                                    }
                                }

                                if (accumulatedGaussian == 0.0f) {
                                    //Reset the gaussian center point to 1.0f. That means
                                    //that for this particular filterData point in the search area
                                    //it is as if the user has opted to include the center point
                                    //value. This avoids the situation where you have a gaussian
                                    //matrix consisting only of zeros:
                                    *(NaNNormalizedGaussian + gaussianCenterOffset) = 1.0f;

                                    //Change the normalization term to 1.0f to reflect the change:
                                    gaussianNormalizationTerm = 1.0f;
                                }
                            }
                            //std::cout<<"GussianNormalizationTerm:"<<GaussianNormalizationTerm<<" ";
                            //Loop through the gaussian buffer and normalize with the above
                            //term (don't bother checking for zeros, just compute):
                            gaussianSum = _mm_load1_ps(&gaussianNormalizationTerm);//Populate vector
                            for (int x = 0; x < nbWindowElts; x += 4)
                            {
                                //Final normalization:
                                gaussianVec =
                                        _mm_div_ps(_mm_load_ps(NaNNormalizedGaussian + x), gaussianSum);

                                //Store:
                                _mm_store_ps((NaNNormalizedGaussian + x), gaussianVec);
                            }

                            //Initialize the iterations sum variable:
                            //IterationsSum = 0.0f;
                            iterationsSum = _mm_load_ss(&zero);

                            //Store the original noisy value (at the center of the background window):
                            originalNoisyValue =
                                    *(unfilteredData + ((top + vPos + edgeWidth) * dataWidth) +
                                      (left + hPos + edgeWidth));

                            //Loop over vectorized window and filter:
                            for (int x = 0; x < nbWindowElts; x += 4)
                            {
                                //Load filterData:
                                filterVec = _mm_load_ps(slidingWindow + x);
                                backgroundVec = _mm_load_ps(bckWindow + x);

                                //Load NaN-normalized / SNR-weighted gaussian:
                                gaussianVec = _mm_load_ps(NaNNormalizedGaussian + x);

                                //compute difference between window and background:
                                result1 = _mm_sub_ps(filterVec, backgroundVec);

                                //Square the difference and multiply by the gaussian:
                                result2 = _mm_mul_ps(gaussianVec, _mm_mul_ps(result1, result1));

                                //Sum result2 across register_s (without using SSE3).
                                //First, store in buffer:
                                _mm_store_ps(register_, result2);
                                temp = 0.0f;

                                //Accumulate:
                                temp +=
                                        (*(register_) + *(register_ + 1) + *(register_ + 2) + *(register_ + 3));

                                //Add iteration sum to running sum:
                                //IterationsSum += Temp;
                                iterationsSum = _mm_add_ss(iterationsSum, _mm_load_ss(&temp));

                            }//End of window element loop.

                            //Next step is to compute the weight value according to
                            //the user selection:

                            if (weightingType == 1)//Standard
                            {
                                float temp;
                                _mm_store_ss(&temp, _mm_mul_ss(coefficient, iterationsSum));
                                weightValue = expf(temp);
                            } else if (weightingType == 2)//Bisquare
                            {
                                //float r = sqrt(IterationsSum);
                                float r;
                                _mm_store_ss(&r, _mm_sqrt_ss(iterationsSum));

                                //compare against the noise parameter value "h":
                                float hCompare;
                                _mm_store_ss(&hCompare, h);

                                if (r <= hCompare) {
                                    //Compute weight via Bisquare method:
                                    //WeightValue =
                                    //	(1.0f - (IterationsSum/hSquared))*(1.0f - (IterationsSum/hSquared));
                                    __m128 term = _mm_sub_ss(one, _mm_div_ss(iterationsSum, hSquared));
                                    _mm_store_ss(&weightValue, _mm_mul_ss(term, term));
                                } else {
                                    //Set to zero:
                                    weightValue = 0.0f;
                                }
                            } else //Modified Bisquare
                            {
                                //float r = sqrt(IterationsSum);
                                float r;
                                _mm_store_ss(&r, _mm_sqrt_ss(iterationsSum));

                                //compare against the noise parameter value "h":
                                float hCompare;
                                _mm_store_ss(&hCompare, h);

                                if (r <= hCompare) {
                                    //Compute weight via Modified Bisquare method:
                                    float term;
                                    _mm_store_ss(&term, _mm_sub_ss(one, _mm_div_ss(iterationsSum, hSquared)));
                                    weightValue = pow(term, 8);
                                } else {
                                    //Set to zero:
                                    weightValue = 0.0f;
                                }
                            }

                            //Check for linear fit option:

                            if (usingLinearFitOption)
                            {
                                std::cout<<"Check for linear regression fit option! not Supported"<<std::endl;


                            }//end of "if UsingLinearFitOption" block.
                            else
                            {
                                //Store to WeightSum:
                                weightValueSum += weightValue;

                                //Compute the ScalarValue3 running total for this filtering position:
                                scalarValue3Sum += (weightValue * originalNoisyValue);
                            }//end of "else averaging" block.

                        }//End of "if CenterValue > 0.0f" block.

                    }//End of "for(int HPosition = 0; ... "
                }//End of "for(int VPosition = 0; ..."


                //check for linear regression fit option:
                if (usingLinearFitOption)
                {
                    std::cout<<"check for linear regression fit option"<<std::endl;

                }//End of "if UsingLinearFit" block.
                else
                {
                    //Compute and store final result:
                    *(filteredData + ((row + edgeWidth) * dataWidth) + (col + edgeWidth)) =
                            (float) (scalarValue3Sum / weightValueSum);
//                    std::cout<<*(FilteredData + ((row + EdgeWidth) * DataWidth) + (Column + EdgeWidth))<<" ";
                }//End of averaging lock.

            }//End of "if (NanMask[][] == 1.0f) && SNR > 0.0f " block.

            else {
                //Store a NaN at this location since there is either
                //a NaN in the original filterData or the SNR is too low:
                *(filteredData + ((row + edgeWidth) * dataWidth) +
                  (col + edgeWidth)) = std::numeric_limits<float>::quiet_NaN();
            }
//            std::cout<<*(FilteredData + ((row + EdgeWidth) * DataWidth) + (Column + EdgeWidth))<<" ";
        }//End of scanline loop (int Column = 0;...).

        //Get the next scanline for processing:
        #pragma omp critical
        {
            currentScanline = synchronizationCounter;
            synchronizationCounter++;
        }

    }//End of filtering loop (while currentScanline <= ...).

    //Deallocate memory here:
    _mm_free(slidingWindow);
    _mm_free(NaNMaskWindow);
    _mm_free(bckWindow);
    _mm_free(gaussianWindow);
    _mm_free(NaNNormalizedGaussian);
    _mm_free(NaNMaskBackground);
    _mm_free(register_);
}


void CNLMFilter::RestoreNaNValsInSNRArray() {
    //Grab a scanline for processing:
    int row;
    #pragma omp critical
    {
        row = synchronizationCounter; // Acquire counter value.
        synchronizationCounter++; // Increment counter.
    }

    while (row < dataHeight) {
        for (int col = 0; col < dataWidth; col++) {
            int currentPosition = (dataWidth * row) + col;

            //Get the filterData value and check for NaN condition:
            if (*(SNRMask + currentPosition) == 0) {
                //NaN at this location. Store NaN in output array:
                *(SNR + currentPosition) = std::numeric_limits<float>::quiet_NaN();
            }
        }

        //Grab the next scanline for processing:
        #pragma omp critical
        {
            row = synchronizationCounter; // Acquire counter value.
            synchronizationCounter++; // Increment counter.
        }
    }
}


void CNLMFilter::PlaceNaNValsInOutputArray() {
    //Scan through the filtered portion of the output array and
    //place a NaN value anywhere there was one originally in the
    //noisy data.
    int offset = (windowSize / 2);//Use int truncation.

    //Grab a scanline for processing:
    int row;
    #pragma omp critical
    {
        row = synchronizationCounter;
        synchronizationCounter++;
    }

    while (row < oHeight) {
        for (int col = 0; col < oWidth; col++) {
            int currentPosition = ((offset + row) * dataWidth) + (offset + col);

            //Get the data value and check for NaN condition:
            if (*(NaNMask + currentPosition) == 0) {
                //NaN at this location. Store NaN in output array:
                *(oData + currentPosition) = std::numeric_limits<float>::quiet_NaN();
            }
        }

        //Grab the next scanline for processing:
        #pragma omp critical
        {
            row = synchronizationCounter;
            synchronizationCounter++;
        }
    }
}

unsigned char *CNLMFilter::GetNaNMask() {
    return NaNMask;
}

void CNLMFilter::FloatFilter_SNR() {
    //Allocate two 1D array of 32-bit floats to hold the filtering window and
    //gaussian values. Buffer this array so that it is a multiple of 4 for SSE:
    __attribute__((aligned(16))) int nbWindowElts = windowSize * windowSize;
    __attribute__((aligned(16))) int bufferedNbVectors = (int) ceil(((float) (windowSize * windowSize)) / 4.0f);

    float *slidingWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4), 16);//Sliding window.
    float *NaNMaskWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4), 16);//Sliding window NaN mask.
    float *SNRSlidingWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4), 16);//SNR filterData.
    float *SNRBackgroundWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4), 16);//SNR filterData.
    float *bckWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4),
                                                   16);//Background values in scanline.
    float *gaussianWindow = (float *) _mm_malloc((bufferedNbVectors * 4 * 4), 16);//Gaussian window.
    float *NaNSNRNormalizedGaussian = (float *) _mm_malloc((bufferedNbVectors * 4 * 4),
                                                           16);//Gaussian normalized for SNR & NaN "holes".
    float *NaNMaskBackground = (float *) _mm_malloc((bufferedNbVectors * 4 * 4),
                                                    16);//NaN mask for background window.
    float *register_ = (float *) _mm_malloc((4 * 4), 16);//For summing across register_.

    //Load the gaussian values from the filterData object to this local array:
    float *gaussianPtr = gaussianWindow;

    for (int x = 0; x < (bufferedNbVectors * 4); x++) {
        *(gaussianWindow + x) = *(gaussianPtr + x);
    }

    //Loop through and load the buffers with zeros:
    for (int x = 0; x < (bufferedNbVectors * 4); x++)
    {
        *(slidingWindow + x) = 0.0f;
        *(NaNMaskWindow + x) = 0.0f;
        *(SNRSlidingWindow + x) = 0.0f;
        *(SNRBackgroundWindow + x) = 0.0f;
        *(bckWindow + x) = 0.0f;
        *(NaNMaskBackground + x) = 0.0f;
        *(NaNSNRNormalizedGaussian + x) = 0.0f;
    }

    //Declare variables to store the runing totals used in the final step
    //of the filtering algorithm:
    __m128 iterationsSum;
    __attribute__((aligned(16))) float temp;
    __attribute__((aligned(16))) float weightValueSum = 0.0f;
    __attribute__((aligned(16))) float weightValue = 0.0f;
    __attribute__((aligned(16))) float scalarValue3Sum = 0.0f;

    //Precompute a coefficient you'll need during the filtering operation:
    __attribute__((aligned(16))) float hOrig = (float) hParam;
    float negOne = -1.0f;
    float two = 2.0f;
    float zero = 0.0f;
    float one_ = 1.0f;
    float hVal = (float) hParam;

    __m128 hOriginal = _mm_load_ss(&hOrig);
    __m128 h = _mm_load_ss(&hVal);
    __m128 hSquared = _mm_mul_ss(h, h);
    __m128 coefficient = _mm_div_ss(_mm_load_ss(&negOne), _mm_mul_ss(hSquared, _mm_load_ss(&two)));

    //Declare some vectors to be used with the scalar SIMD instructions:
    __m128 std = _mm_load_ss(&zero);
    __m128 n = _mm_load_ss(&zero);
    __m128 mean = _mm_load_ss(&zero);
    __m128 one = _mm_load_ss(&one_);


    //Declare variables used to synchronize threads and control the loops:
    __attribute__((aligned(16))) int currentScanline;
    __attribute__((aligned(16))) int totalScanlines = oHeight;
    __attribute__((aligned(16))) int nbHorizontalShifts = oWidth;
    __attribute__((aligned(16))) int edgeWidth = windowSize / 2;//Use integer truncation;
    __attribute__((aligned(16))) int searchEdgeWidth = searchSize / 2;//Use integer truncation.

    //Compute the number of vertical and horizontal shifts the filtering window
    //makes within the search area:
    __attribute__((aligned(16))) int nbHSearchShifts = searchSize - (edgeWidth * 2);
    __attribute__((aligned(16))) int nbVSearchShifts = nbHSearchShifts;

    //Declare variables for the search area limits:
    __attribute__((aligned(16))) int left, top, dataHeight, dataWidth;

    //Some SSE filterData type vectors we'll need:
    __m128 filterVec, backgroundVec, gaussianVec, result1, result2,
            nanWindowVec, snrWindowVec, gaussianSum;


    __attribute__((aligned(16))) float originalNoisyValue;

    //Store the weighting type for use later:
    __attribute__((aligned(16))) int weightingType = weightingFunctionType;

    //Store the linear fit bool:
    bool usingLinearFitOption = false;
    if (useLinearFit > 0) {
        usingLinearFitOption = true;
    }

    __attribute__((aligned(16))) int weightNumberThreshold = minNumberOfWeights;

    //Center of Gaussian array:
    __attribute__((aligned(16))) int gaussianCenterOffset = ((windowSize * windowSize) - 1) / 2;

    //Grab a scanline for processing:
    #pragma omp critical
    {
        currentScanline = synchronizationCounter;
        synchronizationCounter++;
    }


    float *unfilteredData = iData;
    float *filteredData = oData;
    float *SNRData = SNR;

    //Loop over filterData and filter:
    while (currentScanline < totalScanlines)
    {
        //Set the coordinate of the center of the filtering window
        //based on the scanline. A scanline of value 0 starts at
        //the row equal to the edge width in the original filterData array:
        int row = currentScanline;

        //Loop over the scanline, extracting the filter windows
        //and sliding them across the background search area, filtering:

        for (int col = 0; col < nbHorizontalShifts; col++) {
            //Check to see if the current value is NaN, and if
            //the current value has a SNR of 0.0. If not, filter.
            //If so..bypass:
            unsigned char NaNMaskValue = *(NaNMask + ((row + edgeWidth) * dataWidth) + (col + edgeWidth));
            unsigned char SNRMaskValue = *(SNRMask + ((row + edgeWidth) * dataWidth) + (col + edgeWidth));
            float SNRDataValue = *(SNRData + ((row + edgeWidth) * dataWidth) + (col + edgeWidth));

            if ((NaNMaskValue > 0) && (SNRMaskValue > 0) && (SNRDataValue > 0.0f)) {

                //Extract the filtering window and it's associated NaN mask
                // and SNR window based on the row and column centers.
                int increment = 0;
                for (int row = 0; row < windowSize; row++) {
                    for (int column = 0; column < windowSize; column++) {
                        *(slidingWindow + increment) = *(unfilteredData + ((row + row) * dataWidth) +
                                                         (col + column));
                        *(NaNMaskWindow + increment) = (float) (*(NaNMask + ((row + row) * dataWidth) +
                                                                  (col + column)));
                        *(SNRSlidingWindow + increment) = *(SNRData + ((row + row) * dataWidth) + (col + column));
                        increment++;
                    }
                }

                //Check to see if user has selected "adaptive H" option:
                if (useAdaptiveH > 0) {
                    //They are. Reset the adaptive variables:
                    _mm_store_ss(&zero, n);
                    _mm_store_ss(&zero, mean);
                    _mm_store_ss(&zero, std);

                    //Compute the total number of non-NaN values in the current
                    //filtering window by summing over the NaN mask (in the mask, NaN
                    //values are equal to 0, cast to float in NaNMaskWindow):
                    for (int x = 0; x < (windowSize * windowSize); x++) {
                        if (*(NaNMaskWindow + x) > 0.0f) {
                            n = _mm_add_ss(n, one);
                            mean = _mm_add_ss(mean, _mm_load_ss(slidingWindow + x));
                        }
                    }

                    //Compute window mean:
                    mean = _mm_div_ss(mean, _mm_mul_ss(n, n));

                    //Loop through and compute the standard deviation,
                    //again checking for NaNs using the NaN mask:
                    for (int x = 0; x < (windowSize * windowSize); x++) {
                        if (*(NaNMaskWindow + x) > 0.0f) {
                            __m128 value = _mm_load_ss(slidingWindow + x);
                            std = _mm_add_ss(std, _mm_mul_ss(_mm_sub_ss(mean, value), _mm_sub_ss(mean, value)));
                        }
                    }

                    //Final StdDev value for this sliding window:
                    std = _mm_mul_ss(std,_mm_div_ss(one, _mm_sub_ss(_mm_mul_ss(n,n), one)));

                    //Recompute h, hSquared, and coefficient:
                    h = _mm_mul_ss(hOriginal, _mm_sqrt_ss(std));
                    hSquared = _mm_mul_ss(h, h);
                    coefficient =
                            _mm_div_ss(_mm_load_ss(&negOne), _mm_mul_ss(_mm_load_ss(&two), hSquared));

                }//End of "if UsingAdaptiveH" block.

                //Reset the running sum variables to zero:
                weightValueSum = 0.0f;
                scalarValue3Sum = 0.0f;

                //Get the search area limits based on the current filtering pixel.
                //(In hindsight I only needed the left and top values):
                GetSearchLimits(col + edgeWidth, row + edgeWidth, searchSize, dataWidth,
                                dataHeight, &left, &top);

                //Initialize the weight counter and threshold:
                int nbWeightsAboveThreshold = 0;

                //Loop over search area, sliding window and filtering:
                for (int vPos = 0; vPos < nbVSearchShifts; vPos++) {
                    for (int hPos = 0; hPos < nbHSearchShifts; hPos++) {

                        //Check for the presence of a NaN at the center of the
                        //background search patch. Bypass filtering operation
                        //if NaN is present:
                        float centerValue =
                                (float) (*(NaNMask + ((top + vPos + edgeWidth) * dataWidth) +
                                           (left + hPos + edgeWidth)));

                        if (centerValue > 0.0f) {
                            //Extract background window, SNR window and NaN mask into array (NaNMaskWindow
                            //was extracted earlier):
                            increment = 0;
                            for (int row = 0; row < windowSize; row++) {
                                for (int column = 0; column < windowSize; column++) {
                                    *(bckWindow + increment) =
                                            *(unfilteredData + ((top + vPos + row) * dataWidth) +
                                              (left + hPos + column));

                                    *(NaNMaskBackground + increment) =
                                            (float) (*(NaNMask + ((top + vPos + row) * dataWidth) +
                                                       (left + hPos + column)));

                                    *(SNRBackgroundWindow + increment) =
                                            *(SNRData + ((top + vPos + row) * dataWidth) +
                                              (left + hPos + column));

                                    increment++;
                                }
                            }

                            //Normalize the Gaussian kernel to the sum of all the gaussian
                            //values in the kernel that do not correspond to a NaN value in
                            //either the background NaN mask or the window NaN mask. Then,
                            //multiply by the SNR window. If the user is not using SNR
                            //filterData, then the SNR window has been filled with 1.0 earlier.
                            //Use SSE (exclude SSE3 summing across register_ instruction):
                            float gaussianNormalizationTerm = 0.0f;
                            for (int x = 0; x < nbWindowElts; x += 4) {
                                //Combine NaN masks with bitwise AND operation. Just use the
                                //nanWindowVec variable:
                                nanWindowVec =
                                        _mm_and_ps(_mm_load_ps(NaNMaskWindow + x), _mm_load_ps(NaNMaskBackground + x));

                                //Compute product of the sliding window SNR and the background
                                //window SNR:
                                snrWindowVec =
                                        _mm_mul_ps(_mm_load_ps(SNRSlidingWindow + x),
                                                   _mm_load_ps(SNRBackgroundWindow + x));

                                //Multiply the NaN mask and the SNR product, then multiply
                                //this by the Gaussian to filter/weight with NaN/SNR.
                                //Remember - all SNR values are preloaded with 1.0f if user
                                //is not using SNR filterData:
                                gaussianVec =
                                        _mm_mul_ps(_mm_load_ps(gaussianWindow + x),
                                                   _mm_mul_ps(nanWindowVec, snrWindowVec));

                                //Store this result to the gaussian buffer for use later, and
                                //extract and sum (no SSE3) to normalize:
                                _mm_store_ps(register_, gaussianVec);//for accumulation
                                _mm_store_ps((NaNSNRNormalizedGaussian + x), gaussianVec);//For final normalization.

                                //Accumulate:
                                gaussianNormalizationTerm +=
                                        (*(register_) + *(register_ + 1) + *(register_ + 2) + *(register_ + 3));
                            }

                            //Now we have to check to make sure that the gaussian array doesn't consist
                            //of zeros. This is only a concern if the user has opted to exclude the
                            //central value during the processing operation, so check for
                            //that condition first:
                            if (centerPointInclusive < 1) {
                                //They are excluding the center point, so we have to
                                //check the gaussian and make sure all the elements
                                //sum up to more than zero:
                                int increment = 0;
                                float accumulatedGaussian = 0.0f;
                                for (int a = 0; a < windowSize; a++) {
                                    for (int b = 0; b < windowSize; b++) {
                                        accumulatedGaussian += *(NaNSNRNormalizedGaussian + increment);
                                        increment++;
                                    }
                                }

                                if (accumulatedGaussian == 0.0f) {
                                    //Reset the gaussian center point to 1.0f. That means
                                    //that for this particular filterData point in the search area
                                    //it is as if the user has opted to include the center point
                                    //value. This avoids the situation where you have a gaussian
                                    //matrix consisting only of zeros:
                                    *(NaNSNRNormalizedGaussian + gaussianCenterOffset) = 1.0f;

                                    //Change the normalization term to 1.0f to reflect the change:
                                    gaussianNormalizationTerm = 1.0f;
                                }
                            }

                            //Loop through the gaussian buffer and normalize with the above
                            //term (don't bother checking for zeros, just compute):
                            gaussianSum = _mm_load1_ps(&gaussianNormalizationTerm);//Populate vector
                            for (int x = 0; x < nbWindowElts; x += 4) {
                                //Final normalization:
                                gaussianVec =
                                        _mm_div_ps(_mm_load_ps(NaNSNRNormalizedGaussian + x), gaussianSum);

                                //Store:
                                _mm_store_ps((NaNSNRNormalizedGaussian + x), gaussianVec);
                            }

                            //Initialize the iterations sum variable:
                            //IterationsSum = 0.0f;
                            iterationsSum = _mm_load_ss(&zero);

                            //Store the original noisy value (at the center of the background window):
                            originalNoisyValue =
                                    *(unfilteredData + ((top + vPos + edgeWidth) * dataWidth) +
                                      (left + hPos + edgeWidth));

                            //Loop over vectorized window and filter:
                            for (int x = 0; x < nbWindowElts; x += 4) {
                                //Load filterData:
                                filterVec = _mm_load_ps(slidingWindow + x);
                                backgroundVec = _mm_load_ps(bckWindow + x);

                                //Load NaN-normalized / SNR-weighted gaussian:
                                gaussianVec = _mm_load_ps(NaNSNRNormalizedGaussian + x);

                                //compute difference between window and background:
                                result1 = _mm_sub_ps(filterVec, backgroundVec);

                                //Square the difference and multiply by the gaussian:
                                result2 = _mm_mul_ps(gaussianVec, _mm_mul_ps(result1, result1));

                                //Sum result2 across register_s (without using SSE3).
                                //First, store in buffer:
                                _mm_store_ps(register_, result2);
                                temp = 0.0f;

                                //Accumulate:
                                temp +=
                                        (*(register_) + *(register_ + 1) + *(register_ + 2) + *(register_ + 3));

                                //Add iteration sum to running sum:
                                //IterationsSum += Temp;
                                iterationsSum = _mm_add_ss(iterationsSum, _mm_load_ss(&temp));

                            }//End of window element loop.

                            //Next step is to compute the weight value according to
                            //the user selection:

                            if (weightingType == 1)//Standard
                            {
                                float temp;
                                _mm_store_ss(&temp, _mm_mul_ss(coefficient, iterationsSum));
                                weightValue = expf(temp);
                            } else if (weightingType == 2)//Bisquare
                            {
                                //float r = sqrt(IterationsSum);
                                float r;
                                _mm_store_ss(&r, _mm_sqrt_ss(iterationsSum));

                                //compare against the noise parameter value "h":
                                float hCompare;
                                _mm_store_ss(&hCompare, h);

                                if (r <= hCompare) {
                                    //Compute weight via Bisquare method:
                                    __m128 term = _mm_sub_ss(one, _mm_div_ss(iterationsSum, hSquared));
                                    _mm_store_ss(&weightValue, _mm_mul_ss(term, term));
                                } else {
                                    //Set to zero:
                                    weightValue = 0.0f;
                                }
                            } else //Modified Bisquare
                            {
                                //float r = sqrt(IterationsSum);
                                float r;
                                _mm_store_ss(&r, _mm_sqrt_ss(iterationsSum));

                                //compare against the noise parameter value "h":
                                float hCompare;
                                _mm_store_ss(&hCompare, h);

                                if (r <= hCompare) {
                                    //Compute weight via Modified Bisquare method:
                                    float term;
                                    _mm_store_ss(&term, _mm_sub_ss(one, _mm_div_ss(iterationsSum, hSquared)));
                                    weightValue = pow(term, 8);
                                } else {
                                    //Set to zero:
                                    weightValue = 0.0f;
                                }
                            }

                            //Check for linear fit option:
                            if (usingLinearFitOption)
                            {
                                std::cout<< "Linear Regression fit option is not supported !!"<<std::endl;

                            }//end of "if UsingLinearFitOption" block.
                            else {
                                //Store to WeightSum:
                                weightValueSum += weightValue;

                                //Compute the ScalarValue3 running total for this filtering position:
                                scalarValue3Sum += (weightValue * originalNoisyValue);
                            }//end of "else averaging" block.

                        }//End of "if CenterValue > 0.0f" block.

                    }//End of "for(int HPosition = 0; ... "
                }//End of "for(int VPosition = 0; ..."


                //check for linear regression fit option:
                if (usingLinearFitOption)
                {
                    std::cout<< "Linear Regression fit option is not supported !!"<<std::endl;
                }//End of "if UsingLinearFit" block.
                else {
                    //Compute and store final result:
                    *(filteredData + ((row + edgeWidth) * dataWidth) + (col + edgeWidth)) =
                            (float) (scalarValue3Sum / weightValueSum);
                }//End of averaging lock.

            }//End of "if (NanMask[][] == 1.0f) && SNR > 0.0f " block.

            else {
                //Store a NaN at this location since there is either
                //a NaN in the original filterData or the SNR is too low:
                *(filteredData + ((row + edgeWidth) * dataWidth) +
                  (col + edgeWidth)) = std::numeric_limits<float>::quiet_NaN();
            }

        }//End of scanline loop (int Column = 0;...).

        //Get the next scanline for processing:
        #pragma omp critical
        {
            currentScanline = synchronizationCounter;
            synchronizationCounter++;
        }

    }//End of filtering loop (while currentScanline <= ...).

    //Deallocate memory here:
    _mm_free(slidingWindow);
    _mm_free(NaNMaskWindow);
    _mm_free(bckWindow);
    _mm_free(gaussianWindow);
    _mm_free(NaNSNRNormalizedGaussian);
    _mm_free(NaNMaskBackground);
    _mm_free(register_);
    _mm_free(SNRBackgroundWindow);
    _mm_free(SNRSlidingWindow);
}


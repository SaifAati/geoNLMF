#include <float.h>
#include <limits>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <iostream>

#include "geoNLMF.h"

// Constructors:
CNLMFilter::CNLMFilter() {}

void CNLMFilter::SetSearchSize(int size) {
    //Store the search size:
    searchSize = size;
    std::cout<<"SearchSize:"<<searchSize<<std::endl;
}

void CNLMFilter::SetWindowSize(int size) {
    //Store the window size:
    windowSize = size;
    std::cout<<"WindowSize:"<<windowSize<<std::endl;
}

int CNLMFilter::GetWindowSize() {
    return windowSize;
}

void CNLMFilter::StoreWeightingFunctionType(int type) {
    //Store the weighting function type integer:
    weightingFunctionType = type;
    std::cout<< "wightingMethod:"<<weightingFunctionType<<std::endl;
}

void CNLMFilter::StoreMinNumberOfWeights(int number) {
    //Store the minimum number of weights:
    minNumberOfWeights = number;
    std::cout<< "minNumberOfWeights:"<< minNumberOfWeights<<std::endl;
}

void CNLMFilter::StoreMinimumWeightValue(double value) {
    //Store the minimum weight threshold value:
    minimumWeightThreshold = value;
    std::cout<< "minimumWeightThreshold:"<<minimumWeightThreshold<<std::endl;
}

double CNLMFilter::GetMinimumWeightThreshold() {
    // Return the minimum weight threshold value:
    return minimumWeightThreshold;
}

void CNLMFilter::StoreLinearFitOption(int selection) {
    // Linear fi option:
    //  1: For linear regression (Future Work)
    //  2: For gaussian Kernel (Future work)
    //  0: Average Kernel
    useLinearFit = selection;
    std::cout<< "LinearFit:"<<useLinearFit<<std::endl;
}

void CNLMFilter::StoreAdaptiveHoption(int selection) {
    //Store the integer selection:
    useAdaptiveH = selection;
    if (useAdaptiveH==1)
        {std::cout<<"Adaptive H: True"<<std::endl;}
    else {std::cout<<"Adaptive H: False"<<std::endl;}
}

void CNLMFilter::StoreSNRWeightingOption(int selection) {
    //Store the SNR weighting int (yes = 1, no = 0):
    useSNRdata = selection;
    if (useSNRdata==1)
    {std::cout<<"Weight with SNR: True"<<std::endl;}
    else {std::cout<<"Weight with SNR: False"<<std::endl;}
}

bool CNLMFilter::UsesSNRdata() {
    // Return the SNR data selection (true or false):
    bool returnBool;
    if (useSNRdata == 1) { returnBool = true; }
    else { returnBool = false; }

    return returnBool;
}


void CNLMFilter::StoreHparam(double h) {
    //Store the noise estimation:
    hParam = h;
    std::cout<<"Noise H:"<<hParam<<std::endl;
}
void CNLMFilter::StoreCenterPointOption(int selection) {
    //Store the center point option (1 = include, 0 = omit):
    centerPointInclusive = selection;
}



void CNLMFilter::SetDataWidth(int width) {
    //Store the data width:
    dataWidth = width;
}

void CNLMFilter::SetDataHeight(int height) {
    //Store data height:
    dataHeight = height;
}

int CNLMFilter::GetDataHeight() {
    return dataHeight;
}

int CNLMFilter::GetDataWidth() {
    return dataWidth;
}

void CNLMFilter::ComputeOutputDimensions() {
    //compute the dimensions of the output array minus the
    //unfiltered edges:
    int edgeWidth = windowSize / 2; //Use int truncation.
    oHeight = dataHeight - (2 * edgeWidth);
    oWidth = dataWidth - (2 * edgeWidth);
    std::cout<<"oDim:"<< oHeight<<","<<oWidth<<std::endl;
}

void CNLMFilter::AllocateNaNMask() {
    //Allocate the NaNMask (align on 16-byte boundary):
    NaNMask = (unsigned char *) _mm_malloc((dataWidth * dataHeight), 16);
}

void CNLMFilter::StoreInputArrayPtr(float *inputPtr)
{
    //Store the input array pointer:
    iData = inputPtr;
    //std::cout<<*(InputData_F+245)<<std::endl;
}
void CNLMFilter::StoreOutputArrayPtr(float *outputPtr) {
    //Store the output array pointer:
    oData = outputPtr;
//    std::cout<<*(OutputData_F+1)<<std::endl;
}


void CNLMFilter::ComputeGaussianWindow() {
    //NOTE: The normalization of the gaussian occurs in the filtering thread
    //function in order to normalize a gaussian that has values which spatially
    //correspond to NaN values in the original unfiltered data.

    //Calculate the StdDev based on the user-input window size:
    float sigma = (((float) windowSize) - 1.0f) / 2.0f;
    std::cout<<"Sigma:"<<sigma<<" ";
    //Allocate a 2d (it's easier to compute) float array to hold the
    //gaussian kernel filter coefficients:
    float **GaussianKernel;
    GaussianKernel = new float *[windowSize];//Rows.
    for (int row = 0; row < windowSize; row++) {
        GaussianKernel[row] = new float[windowSize];//Columns.
    }

    //Loop through the array, compute and store the Gaussian coefficients:
    int WindowCenter = windowSize / 2;

    for (int row = 0; row < windowSize; row++) {
        for (int column = 0; column < windowSize; column++) {
            //Compute distance from center of kernel window:
            float RowDistance = (float) (WindowCenter - row);
            float ColumnDistance = (float) (WindowCenter - column);
            float Distance = sqrt((RowDistance * RowDistance) + (ColumnDistance * ColumnDistance));

            //Compute coefficient based on distance and StdDev:
            GaussianKernel[row][column] = expf(-0.5f * ((Distance / sigma) * (Distance / sigma)));
        }
    }

    //Check the center point included option. If excluded,
    //set the inner gaussian value to zero:
    if (centerPointInclusive == 0) {
        int center = windowSize / 2;//Use integer truncation.
        GaussianKernel[center][center] = 0.0f;
    }

    //Allocate the 1D memory array, aligned on 16-byte boundary. This
    //consists of the gaussian coefficients plus the zero buffering needed
    //for the vectorization. Each vector consists of 4 floats, so to
    //compute the buffer size, divide the total number of window elements
    //by 4 and round up to the next integer:
    float NumberOfBufferedElements = (float) (windowSize * windowSize);
    int NumberOfVectors = (int) (ceil(NumberOfBufferedElements / 4.0f));

    gaussianWindow = (float *) _mm_malloc((NumberOfVectors * 4 * 4), 16);

    //Loop through and load buffer with zeros:
    int Position = 0;
    for (int x = 0; x < (NumberOfVectors * 4); x++) {
        *(gaussianWindow + Position) = 0.0f;
        Position++;
    }
    //Loop through and load the gaussian values:
    Position = 0;
    for (int row = 0; row < windowSize; row++)//"column" is correct.
    {
        for (int column = 0; column < windowSize; column++)//"row" is correct.
        {
            *(gaussianWindow + Position) = GaussianKernel[row][column];
            Position++;
        }
    }


    //Deallocate temporary GaussianKernel memory:
    for (int x = 0; x < windowSize; x++) {
        delete[] GaussianKernel[x];
    }
    delete[] GaussianKernel;
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

    //Store the SNR array pointer:
    SNR = snrPtr;
}


void CNLMFilter::AllocateSNRMask() {
    //Allocate the SNRMask (align on 16-byte boundary):
    SNRMask = (unsigned char *) _mm_malloc((dataWidth * dataHeight), 16);

}

void CNLMFilter::GetSearchLimits(int XPos, int YPos, int SearchDimension, int dataWidth,
                                 int dataHeight, int *left, int *top) {
    //We need to determine the physical extents of the search area
    //i.e. the coords in the data matrix (actually only the left and top) based
    //on the current position in the data matrix and the edge dimension
    //of the user-specified search area. There are 9 possibilities.
    //Use a series of "if-else" statements to determine which one we
    //are dealing with and set the search limits. Note: ALL COORDS
    //ARE DATA ARRAY COORDS!:

    int SearchEdgeDim = SearchDimension / 2;//Use int truncation.

    if ((XPos <= SearchEdgeDim) && (YPos <= SearchEdgeDim))//Possibility 1
    {
        //The upper left corner up to and including the first
        //position wherein the search area will increment along
        //with the X and Y pos.
        *(left) = 0;
        *(top) = 0;
    } else if (((XPos > SearchEdgeDim) && (XPos < (dataWidth - SearchEdgeDim - 1))) &&
               (YPos <= SearchEdgeDim))//Possibility 2
    {
        //The upper middle portion wherein the search area slides
        //horizontally, but not vertically, with the filtering window:
        *(left) = (XPos - SearchEdgeDim);
        *(top) = 0;
    } else if ((XPos >= (dataWidth - SearchEdgeDim - 1)) && (YPos <= SearchEdgeDim))//Possibility 3
    {
        //The upper right corner where the search window cannot slide
        //with the filtering window:
        *(left) = (dataWidth - SearchEdgeDim * 2 - 1);
        *(top) = 0;
    } else if ((XPos <= SearchEdgeDim) && ((YPos > SearchEdgeDim) &&
                                           (YPos < (dataHeight - SearchEdgeDim - 1))))//Possibility 4.
    {
        //The middle left portion where the search window increments vertically
        //but not horizontally with the sliding filter window:
        *(left) = 0;
        *(top) = (YPos - SearchEdgeDim);
    } else if (((XPos > SearchEdgeDim) && (XPos < (dataWidth - SearchEdgeDim - 1))) &&
               ((YPos > SearchEdgeDim) && (YPos < (dataHeight - SearchEdgeDim - 1))))//Possibility 5.
    {
        //The middle of the data array, where the search window shifts with the filtering
        //window.
        *(left) = (XPos - SearchEdgeDim);
        *(top) = (YPos - SearchEdgeDim);
    } else if ((XPos >= (dataWidth - SearchEdgeDim - 1)) &&
               ((YPos > SearchEdgeDim) && (YPos < (dataHeight - SearchEdgeDim - 1))))//Possibility 6.
    {
        //The middle right-hand portion of the array, where the search area can increment
        //vertically but not horizontally:
        *(left) = (dataWidth - SearchEdgeDim * 2 - 1);
        *(top) = (YPos - SearchEdgeDim);
    } else if ((XPos <= SearchEdgeDim) && (YPos >= (dataHeight - SearchEdgeDim - 1)))//Possibility 7
    {
        //The lower left hand portion of the array, where the search area cannot increment
        //with the filtering window:
        *(left) = 0;
        *(top) = (dataHeight - SearchEdgeDim * 2 - 1);
    } else if (((XPos > SearchEdgeDim) && (XPos < (dataWidth - SearchEdgeDim - 1))) &&
               (YPos >= (dataHeight - SearchEdgeDim - 1)))//Possibility 8
    {
        //The bottom middle portion of the data array, where the search area can increment
        //horizontally with the filter window but not vertically:
        *(left) = (XPos - SearchEdgeDim);
        *(top) = (dataHeight - SearchEdgeDim * 2 - 1);
    } else//Possibility 9
    {
        //The lower right portion of the data array, where the search area cannot
        //increment with the filtering window:
        *(left) = (dataWidth - SearchEdgeDim * 2 - 1);
        *(top) = (dataHeight - SearchEdgeDim * 2 - 1);
    }
}

void CNLMFilter::GenerateNaNMask() {
    //Acquire a scanline for processing:
    int Row;
    #pragma omp critical
    {
        Row = synchronizationCounter; // Acquire counter value.
        synchronizationCounter++; // Increment counter.
    }
    while (Row < dataHeight) {
        for (int Column = 0; Column < dataWidth; Column++) {
            //Get the filterData value and check for NaN condition:
            if (__isnan(iData[(Row * dataWidth) + Column])) {
                //Value is NaN. Set the value in the mask to 0:
                NaNMask[(Row * dataWidth) + Column] = 0;

                //Replace the unfiltered NaN with zero:
                iData[(Row * dataWidth) + Column] = 0.0f;
            } else {
                NaNMask[(Row * dataWidth) + Column] = 1;
            }
        }

        //Acquire another scanline for processing:
    #pragma omp critical
        {
            Row = synchronizationCounter; // Acquire counter value.
            synchronizationCounter++; // Increment counter.
        }
    }
}


void CNLMFilter::ReplaceNaNValsInSNRArray() {
    //Grab a scanline for processing:
    int Row;
    #pragma omp critical
    {
        Row = synchronizationCounter;
        synchronizationCounter++;
    }

    while (Row < dataHeight) {
        for (int Column = 0; Column < dataWidth; Column++) {
            //Get the filterData value and check for NaN condition:
            if (__isnan(SNR[(Row * dataWidth) + Column])) {
                //Value is NaN. Replace the value with 0.0f:
                SNR[(Row * dataWidth) + Column] = 0.0f;

                //Set the mask with 0 to indicate presence of a NaN:
                SNRMask[(Row * dataWidth) + Column] = 0;
            } else {
                //Set the value in the mask to 1:
                SNRMask[(Row * dataWidth) + Column] = 1;
            }
        }

        //Grab the next scanline for processing:
    #pragma omp critical
        {
            Row = synchronizationCounter;
            synchronizationCounter++;
        }
    }
}

void CNLMFilter::Filter() {

    //Allocate two 1D array of 32-bit floats to hold the filtering window and
    //gaussian values. Buffer this array so that it is a multiple of 4 for SSE:
    int NumberOfWindowElements = windowSize * windowSize;
    __attribute__((aligned(16))) int BufferedNumberOfVectors = (int) ceil(((float) (windowSize * windowSize)) / 4.0f);

    float *SlidingWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4), 16);//Sliding window.
    float *NaNMaskWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4), 16);//Sliding window NaN mask.
    float *BackgroundWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4),16);//Background values in scanline.
    float *GaussianWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4), 16);//Gaussian window.
    float *NaNNormalizedGaussian = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4),
                                                        16);//Gaussian normalized for SNR & NaN "holes".
    float *NaNMaskBackground = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4),
                                                    16);//NaN mask for background window.
    float *Register = (float *) _mm_malloc((4 * 4), 16);//For summing across register.

    //Load the gaussian values from the filterData object to this local array:
    float *GaussianPtr = gaussianWindow;

    for (int x = 0; x < (BufferedNumberOfVectors * 4); x++)
    {
        *(GaussianWindow + x) = *(GaussianPtr + x);
    }

    //Loop through and load the buffers with zeros:
    for (int x = 0; x < (BufferedNumberOfVectors * 4); x++)
    {
        *(SlidingWindow + x) = 0.0f;
        *(NaNMaskWindow + x) = 0.0f;
        *(BackgroundWindow + x) = 0.0f;
        *(NaNMaskBackground + x) = 0.0f;
        *(NaNNormalizedGaussian + x) = 0.0f;
    }

    //Declare variables to store the running totals used in the final step of the filtering algorithm:
    __m128 IterationsSum;

    __attribute__((aligned(16))) float Temp;
    __attribute__((aligned(16))) float WeightValueSum = 0.0f;
    __attribute__((aligned(16))) float WeightValue = 0.0f;
    __attribute__((aligned(16))) float ScalarValue3Sum = 0.0f;

    //Precompute a coefficient you'll need during the filtering operation:
    __attribute__((aligned(16))) float hOrig = (float) hParam;
    float NegOne = -1.0f;
    float Two = 2.0f;
    float Zero = 0.0f;
    float one = 1.0f;
    float Hval = (float) hParam;

    __m128 hOriginal = _mm_load_ss(&hOrig);
    __m128 h = _mm_load_ss(&Hval);
    __m128 hSquared = _mm_mul_ss(h, h);
    __m128 coefficient = _mm_div_ss(_mm_load_ss(&NegOne), _mm_mul_ss(hSquared, _mm_load_ss(&Two)));

    //Declare some vectors to be used with the scalar SIMD instructions:

    __m128 StandardDeviation = _mm_load_ss(&Zero);
    __m128 N = _mm_load_ss(&Zero);
    __m128 Mean = _mm_load_ss(&Zero);
    __m128 One = _mm_load_ss(&one);


    //Declare variables used to synchronize threads and control the loops:
    __attribute__((aligned(16))) int CurrentScanline;
    __attribute__((aligned(16))) int TotalScanlines = oHeight;
    __attribute__((aligned(16))) int NumberOfHorizontalShifts = oWidth;
    __attribute__((aligned(16))) int EdgeWidth = windowSize / 2;//Use integer truncation;
    __attribute__((aligned(16))) int SearchEdgeWidth = searchSize / 2;//Use integer truncation.

    //Compute the number of vertical and horizontal shifts the filtering window
    //makes within the search area:
    __attribute__((aligned(16))) int NumberOfHSearchShifts = searchSize - (EdgeWidth * 2);
    __attribute__((aligned(16))) int NumberOfVSearchShifts = NumberOfHSearchShifts;

    //Declare variables for the search area limits:
    __attribute__((aligned(16))) int Left, Top;

    //Some SSE filterData type vectors we'll need:
    __m128 filterVec, backgroundVec, gaussianVec, result1, result2,
            nanWindowVec, gaussianSum;

//    //Used for the SV-decomposition solution of linear equations in the Intel IPP library:
//    double A[3 * 3];
//    double U[3 * 3];
//    double U_transposed[3 * 3];
//    double V[3 * 3];
//    double W[3];
//    double Wt[3 * 3];
//    double Temp1[3 * 3];
//    double Temp2[3 * 3];
//    double B[3];
//    double X[3];
//    int srcStride2_64 = sizeof(double);
//    int srcStride1_64 = 3 * sizeof(double);
//    int decompStride1 = 3 * sizeof(float);
//    int decompStride2 = sizeof(float);

//    for (int x = 0; x < 9; x++)
//    {
//        Wt[x] = 0.0;
//    }

    __attribute__((aligned(16))) float OriginalNoisyValue;

    //Store the weighting type for use later:
    __attribute__((aligned(16))) int WeightingType = weightingFunctionType;

    //Store the linear fit bool:
    bool UsingLinearFitOption = false;
    if (useLinearFit > 0)
    {
        UsingLinearFitOption = true;
        std::cout<<"Using Linear Fit Option: "<<UsingLinearFitOption<<std::endl;

    }

    //__declspec(align(16)) float MinimumWeightThreshold = (float)MinimumWeightThreshold;
    __attribute__((aligned(16))) int WeightNumberThreshold = minNumberOfWeights;

    //Center of Gaussian array:
    __attribute__((aligned(16))) int GaussianCenterOffset = ((windowSize * windowSize) - 1) / 2;

    //Grab a scanline for processing:
    #pragma omp critical
    {
        CurrentScanline = synchronizationCounter;
        synchronizationCounter++;
    }

    //unsigned char* NaNMask = NaNMask;
    float *UnfilteredData = iData;
    float *FilteredData = oData;


    //Loop over filterData and filter:
//    std::cout<< "TotalScanlines:"<<TotalScanlines<<std::endl;
//    TotalScanlines = 1;
    while (CurrentScanline < TotalScanlines)
    {
        //Set the coordinate of the center of the filtering window
        //based on the scanline. A scanline of value 0 starts at
        //the row equal to the edge width in the original filterData array:
        int Row = CurrentScanline;

        //Loop over the scanline, extracting the filter windows
        //and sliding them across the background search area, filtering:

        for (int Column = 0; Column < NumberOfHorizontalShifts; Column++)
        {
            //Check to see if the current value is NaN.
            //If so..bypass:
            unsigned char NaNMaskValue = *(NaNMask + ((Row + EdgeWidth) * dataWidth) + (Column + EdgeWidth));

            if (NaNMaskValue > 0)
            {
                //Extract the filtering window and it's associated NaN mask
                // and SNR window based on the row and column centers.
                int increment = 0;
                for (int row = 0; row < windowSize; row++)
                {
                    for (int column = 0; column < windowSize; column++) {
                        *(SlidingWindow + increment) = *(UnfilteredData + ((Row + row) * dataWidth) +
                                                         (Column + column));
                        *(NaNMaskWindow + increment) = (float) (*(NaNMask + ((Row + row) * dataWidth) +
                                                                  (Column + column)));
                        increment++;
                    }
                }
//
                //Check to see if user has selected "adaptive H" option:
                if (useAdaptiveH > 0)
                {
                    //They are. Reset the adaptive variables:
                    //N                 = 0.0f;
                    //Mean              = 0.0f;
                    //StandardDeviation = 0.0f;
                    _mm_store_ss(&Zero, N);
                    _mm_store_ss(&Zero, Mean);
                    _mm_store_ss(&Zero, StandardDeviation);

                    //Compute the total number of non-NaN values in the current
                    //filtering window by summing over the NaN mask (in the mask, NaN
                    //values are equal to 0, cast to float in NaNMaskWindow):
                    for (int x = 0; x < (windowSize * windowSize); x++) {
                        if (*(NaNMaskWindow + x) > 0.0f) {
                            //N += 1.0f;//Number of non-NaNs.
                            //Mean += *(SlidingWindow + x);//Running total.
                            N = _mm_add_ss(N, One);
                            Mean = _mm_add_ss(Mean, _mm_load_ss(SlidingWindow + x));
                        }
                    }

                    //Compute window mean:
                    //Mean = Mean / (N*N);
                    Mean = _mm_div_ss(Mean, _mm_mul_ss(N, N));

                    //Loop through and compute the standard deviation,
                    //again checking for NaNs using the NaN mask:
                    for (int x = 0; x < (windowSize * windowSize); x++) {
                        if (*(NaNMaskWindow + x) > 0.0f) {
                            //float value = *(SlidingWindow + x);
                            //StandardDeviation += (Mean - value)*(Mean - value);
                            __m128 value = _mm_load_ss(SlidingWindow + x);
                            StandardDeviation =
                                    _mm_add_ss(StandardDeviation,
                                               _mm_mul_ss(_mm_sub_ss(Mean, value), _mm_sub_ss(Mean, value)));
                        }
                    }

                    //Final StdDev value for this sliding window:
                    //StandardDeviation = StandardDeviation * (1.0f / (N*N - 1.0f));
                    StandardDeviation = _mm_mul_ss(StandardDeviation,
                                                   _mm_div_ss(One, _mm_sub_ss(_mm_mul_ss(N, N), One)));

                    //Recompute h, hSquared, and coefficient:
                    //h           = hOriginal * (sqrt(StandardDeviation));
                    //hSquared    = h * h;
                    //coefficient = -1.0f / (2.0f * (hSquared));
                    h = _mm_mul_ss(hOriginal, _mm_sqrt_ss(StandardDeviation));
                    hSquared = _mm_mul_ss(h, h);
                    coefficient =
                            _mm_div_ss(_mm_load_ss(&NegOne), _mm_mul_ss(_mm_load_ss(&Two), hSquared));

                }//End of "if UsingAdaptiveH" block.

                //Reset the running sum variables to zero:
                WeightValueSum = 0.0f;
                ScalarValue3Sum = 0.0f;

                //Get the search area limits based on the current filtering pixel.
                //(In hindsight I only needed the left and top values):
                GetSearchLimits(Column + EdgeWidth, Row + EdgeWidth, searchSize, dataWidth,
                                dataHeight, &Left, &Top);
                //Loop over search area, sliding window and filtering:
                for (int VPosition = 0; VPosition < NumberOfVSearchShifts; VPosition++)
                {
                    for (int HPosition = 0; HPosition < NumberOfHSearchShifts; HPosition++)
                    {

                        //Check for the presence of a NaN at the center of the
                        //background search patch. Bypass filtering operation
                        //if NaN is present:
                        float CenterValue =
                                (float) (*(NaNMask + ((Top + VPosition + EdgeWidth) * dataWidth) +
                                           (Left + HPosition + EdgeWidth)));
                        if (CenterValue > 0.0f)
                        {
                            increment = 0;
                            for (int row = 0; row < windowSize; row++)
                            {
                                for (int column = 0; column < windowSize; column++) {
                                    *(BackgroundWindow + increment) =
                                            *(UnfilteredData + ((Top + VPosition + row) * dataWidth) +
                                              (Left + HPosition + column));

                                    *(NaNMaskBackground + increment) =
                                            (float) (*(NaNMask + ((Top + VPosition + row) * dataWidth) +
                                                       (Left + HPosition + column)));

                                    increment++;
                                }
                            }

                            //Normalize the Gaussian kernel to the sum of all the gaussian
                            //values in the kernel that do not correspond to a NaN value in
                            //either the background NaN mask or the window NaN mask.
                            //Use SSE (exclude SSE3 summing across register instruction):
                            float GaussianNormalizationTerm = 0.0f;
                            for (int x = 0; x < NumberOfWindowElements; x += 4)
                            {
                                //Combine NaN masks with bitwise AND operation. Just use the
                                //nanWindowVec variable:
                                nanWindowVec =
                                        _mm_and_ps(_mm_load_ps(NaNMaskWindow + x), _mm_load_ps(NaNMaskBackground + x));

                                //Multiply the Gaussian by the NaN mask to filter with NaN:
                                gaussianVec =
                                        _mm_mul_ps(_mm_load_ps(GaussianWindow + x), nanWindowVec);

                                //Store this result to the gaussian buffer for use later, and
                                //extract and sum (no SSE3) to normalize:
                                _mm_store_ps(Register, gaussianVec);//for accumulation
                                _mm_store_ps((NaNNormalizedGaussian + x), gaussianVec);//For final normalization.

                                //Accumulate:
                                GaussianNormalizationTerm +=
                                        (*(Register) + *(Register + 1) + *(Register + 2) + *(Register + 3));
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
                                float AccumulatedGaussian = 0.0f;
                                for (int a = 0; a < windowSize; a++) {
                                    for (int b = 0; b < windowSize; b++) {
                                        AccumulatedGaussian += *(NaNNormalizedGaussian + increment);
                                        increment++;
                                    }
                                }

                                if (AccumulatedGaussian == 0.0f) {
                                    //Reset the gaussian center point to 1.0f. That means
                                    //that for this particular filterData point in the search area
                                    //it is as if the user has opted to include the center point
                                    //value. This avoids the situation where you have a gaussian
                                    //matrix consisting only of zeros:
                                    *(NaNNormalizedGaussian + GaussianCenterOffset) = 1.0f;

                                    //Change the normalization term to 1.0f to reflect the change:
                                    GaussianNormalizationTerm = 1.0f;
                                }
                            }
                            //std::cout<<"GussianNormalizationTerm:"<<GaussianNormalizationTerm<<" ";
                            //Loop through the gaussian buffer and normalize with the above
                            //term (don't bother checking for zeros, just compute):
                            gaussianSum = _mm_load1_ps(&GaussianNormalizationTerm);//Populate vector
                            for (int x = 0; x < NumberOfWindowElements; x += 4)
                            {
                                //Final normalization:
                                gaussianVec =
                                        _mm_div_ps(_mm_load_ps(NaNNormalizedGaussian + x), gaussianSum);

                                //Store:
                                _mm_store_ps((NaNNormalizedGaussian + x), gaussianVec);
                            }

                            //Initialize the iterations sum variable:
                            //IterationsSum = 0.0f;
                            IterationsSum = _mm_load_ss(&Zero);

                            //Store the original noisy value (at the center of the background window):
                            OriginalNoisyValue =
                                    *(UnfilteredData + ((Top + VPosition + EdgeWidth) * dataWidth) +
                                      (Left + HPosition + EdgeWidth));

                            //Loop over vectorized window and filter:
                            for (int x = 0; x < NumberOfWindowElements; x += 4)
                            {
                                //Load filterData:
                                filterVec = _mm_load_ps(SlidingWindow + x);
                                backgroundVec = _mm_load_ps(BackgroundWindow + x);

                                //Load NaN-normalized / SNR-weighted gaussian:
                                gaussianVec = _mm_load_ps(NaNNormalizedGaussian + x);

                                //compute difference between window and background:
                                result1 = _mm_sub_ps(filterVec, backgroundVec);

                                //Square the difference and multiply by the gaussian:
                                result2 = _mm_mul_ps(gaussianVec, _mm_mul_ps(result1, result1));

                                //Sum result2 across registers (without using SSE3).
                                //First, store in buffer:
                                _mm_store_ps(Register, result2);
                                Temp = 0.0f;

                                //Accumulate:
                                Temp +=
                                        (*(Register) + *(Register + 1) + *(Register + 2) + *(Register + 3));

                                //Add iteration sum to running sum:
                                //IterationsSum += Temp;
                                IterationsSum = _mm_add_ss(IterationsSum, _mm_load_ss(&Temp));

                            }//End of window element loop.

                            //Next step is to compute the weight value according to
                            //the user selection:

                            if (WeightingType == 1)//Standard
                            {
                                float temp;
                                _mm_store_ss(&temp, _mm_mul_ss(coefficient, IterationsSum));
                                WeightValue = expf(temp);
                            } else if (WeightingType == 2)//Bisquare
                            {

                                float r;
                                _mm_store_ss(&r, _mm_sqrt_ss(IterationsSum));

                                //compare against the noise parameter value "h":
                                float hCompare;
                                _mm_store_ss(&hCompare, h);

                                if (r <= hCompare) {
                                    //Compute weight via Bisquare method:
                                    //WeightValue =
                                    //	(1.0f - (IterationsSum/hSquared))*(1.0f - (IterationsSum/hSquared));
                                    __m128 Term = _mm_sub_ss(One, _mm_div_ss(IterationsSum, hSquared));
                                    _mm_store_ss(&WeightValue, _mm_mul_ss(Term, Term));
                                } else {
                                    //Set to zero:
                                    WeightValue = 0.0f;
                                }
                            } else //Modified Bisquare
                            {
                                //float r = sqrt(IterationsSum);
                                float r;
                                _mm_store_ss(&r, _mm_sqrt_ss(IterationsSum));

                                //compare against the noise parameter value "h":
                                float hCompare;
                                _mm_store_ss(&hCompare, h);

                                if (r <= hCompare) {
                                    //Compute weight via Modified Bisquare method:
                                    float Term;
                                    _mm_store_ss(&Term, _mm_sub_ss(One, _mm_div_ss(IterationsSum, hSquared)));
                                    WeightValue = pow(Term, 8);
                                } else {
                                    //Set to zero:
                                    WeightValue = 0.0f;
                                }
                            }

                            //Check for linear fit option:
                            if (UsingLinearFitOption)
                            {
                            }//end of "if UsingLinearFitOption" block.
                            else
                            {
                                //Store to WeightSum:
                                WeightValueSum += WeightValue;

                                //Compute the ScalarValue3 running total for this filtering position:
                                ScalarValue3Sum += (WeightValue * OriginalNoisyValue);
                            }//end of "else averaging" block.
                        }//End of "if CenterValue > 0.0f" block.
                    }//End of "for(int HPosition = 0; ... "
                }//End of "for(int VPosition = 0; ..."

                //check for linear regression fit option:
                if (UsingLinearFitOption)
                {
                    std::cout<<"check for linear regression fit option"<<std::endl;
                    std::cout<<"UsingLinearFitOption:Error with ipp SVD decomposition  "<<std::endl;
                }//End of "if UsingLinearFit" block.
                else
                {
                    //Compute and store final result:
                    *(FilteredData + ((Row + EdgeWidth) * dataWidth) + (Column + EdgeWidth)) =
                            (float) (ScalarValue3Sum / WeightValueSum);
                }//End of averaging lock.
            }//End of "if (NanMask[][] == 1.0f) && SNR > 0.0f " block.

            else {
                //Store a NaN at this location since there is either
                //a NaN in the original filterData or the SNR is too low:
                *(FilteredData + ((Row + EdgeWidth) * dataWidth) +
                  (Column + EdgeWidth)) = std::numeric_limits<float>::quiet_NaN();
            }
        }//End of scanline loop (int Column = 0;...).

        //Get the next scanline for processing:
        #pragma omp critical
        {
            CurrentScanline = synchronizationCounter;
            synchronizationCounter++;
        }

    }//End of filtering loop (while CurrentScanline <= ...).

    //Deallocate memory here:
    _mm_free(SlidingWindow);
    _mm_free(NaNMaskWindow);
    _mm_free(BackgroundWindow);
    _mm_free(GaussianWindow);
    _mm_free(NaNNormalizedGaussian);
    _mm_free(NaNMaskBackground);
    _mm_free(Register);
}


void CNLMFilter::RestoreNaNValsInSNRArray() {
    //Grab a scanline for processing:
    int Row;
#pragma omp critical
    {
        Row = synchronizationCounter; // Acquire counter value.
        synchronizationCounter++; // Increment counter.
    }

    while (Row < dataHeight) {
        for (int Column = 0; Column < dataWidth; Column++) {
            int CurrentPosition = (dataWidth * Row) + Column;

            //Get the filterData value and check for NaN condition:
            if (*(SNRMask + CurrentPosition) == 0) {
                //NaN at this location. Store NaN in output array:
                *(SNR + CurrentPosition) = std::numeric_limits<float>::quiet_NaN();
            }
        }

        //Grab the next scanline for processing:
        #pragma omp critical
        {
            Row = synchronizationCounter; // Acquire counter value.
            synchronizationCounter++; // Increment counter.
        }
    }
}


void CNLMFilter::PlaceNaNValsInOutputArray() {
    //Scan through the filtered portion of the output array and
    //place a NaN value anywhere there was one originally in the
    //noisy data.
    int Offset = (windowSize / 2);//Use int truncation.

    //Grab a scanline for processing:
    int Row;
#pragma omp critical
    {
        Row = synchronizationCounter;
        synchronizationCounter++;
    }

    while (Row < oHeight) {
        for (int Column = 0; Column < oWidth; Column++) {
            int CurrentPosition = ((Offset + Row) * dataWidth) + (Offset + Column);

            //Get the data value and check for NaN condition:
            if (*(NaNMask + CurrentPosition) == 0) {
                //NaN at this location. Store NaN in output array:
                *(oData + CurrentPosition) = std::numeric_limits<float>::quiet_NaN();
            }
        }

        //Grab the next scanline for processing:
#pragma omp critical
        {
            Row = synchronizationCounter;
            synchronizationCounter++;
        }
    }
}

unsigned char *CNLMFilter::GetNaNMask() {
    return NaNMask;
}



void CNLMFilter::Filter_SNR() {
    //Allocate two 1D array of 32-bit floats to hold the filtering window and
    //gaussian values. Buffer this array so that it is a multiple of 4 for SSE:
    __attribute__((aligned(16))) int NumberOfWindowElements = windowSize * windowSize;
    __attribute__((aligned(16))) int BufferedNumberOfVectors = (int) ceil(((float) (windowSize * windowSize)) / 4.0f);

    float *SlidingWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4), 16);//Sliding window.
    float *NaNMaskWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4), 16);//Sliding window NaN mask.
    float *SNRSlidingWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4), 16);//SNR filterData.
    float *SNRBackgroundWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4), 16);//SNR filterData.
    float *BackgroundWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4),
                                                   16);//Background values in scanline.
    float *GaussianWindow = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4), 16);//Gaussian window.
    float *NaNSNRNormalizedGaussian = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4),
                                                           16);//Gaussian normalized for SNR & NaN "holes".
    float *NaNMaskBackground = (float *) _mm_malloc((BufferedNumberOfVectors * 4 * 4),
                                                    16);//NaN mask for background window.
    float *Register = (float *) _mm_malloc((4 * 4), 16);//For summing across register.

    //Load the gaussian values from the filterData object to this local array:
    float *GaussianPtr = gaussianWindow;

    for (int x = 0; x < (BufferedNumberOfVectors * 4); x++) {
        *(GaussianWindow + x) = *(GaussianPtr + x);
    }

    //Loop through and load the buffers with zeros:
    for (int x = 0; x < (BufferedNumberOfVectors * 4); x++)
    {
        *(SlidingWindow + x) = 0.0f;
        *(NaNMaskWindow + x) = 0.0f;
        *(SNRSlidingWindow + x) = 0.0f;
        *(SNRBackgroundWindow + x) = 0.0f;
        *(BackgroundWindow + x) = 0.0f;
        *(NaNMaskBackground + x) = 0.0f;
        *(NaNSNRNormalizedGaussian + x) = 0.0f;
    }

    //Declare variables to store the runing totals used in the final step
    //of the filtering algorithm:
    //__declspec(align(16)) float IterationsSum;
    __m128 IterationsSum;
    __attribute__((aligned(16))) float Temp;
    __attribute__((aligned(16))) float WeightValueSum = 0.0f;
    __attribute__((aligned(16))) float WeightValue = 0.0f;
    __attribute__((aligned(16))) float ScalarValue3Sum = 0.0f;

    //Precompute a coefficient you'll need during the filtering operation:
    __attribute__((aligned(16))) float hOrig = (float) hParam;
    float NegOne = -1.0f;
    float Two = 2.0f;
    float Zero = 0.0f;
    float one = 1.0f;
    float Hval = (float) hParam;

    __m128 hOriginal = _mm_load_ss(&hOrig);
    __m128 h = _mm_load_ss(&Hval);
    __m128 hSquared = _mm_mul_ss(h, h);
    __m128 coefficient = _mm_div_ss(_mm_load_ss(&NegOne), _mm_mul_ss(hSquared, _mm_load_ss(&Two)));

    //Declare some vectors to be used with the scalar SIMD instructions:
    __m128 StandardDeviation = _mm_load_ss(&Zero);
    __m128 N = _mm_load_ss(&Zero);
    __m128 Mean = _mm_load_ss(&Zero);
    __m128 One = _mm_load_ss(&one);


    //Declare variables used to synchronize threads and control the loops:
    __attribute__((aligned(16))) int CurrentScanline;
    __attribute__((aligned(16))) int TotalScanlines = oHeight;
    __attribute__((aligned(16))) int NumberOfHorizontalShifts = oWidth;
    __attribute__((aligned(16))) int EdgeWidth = windowSize / 2;//Use integer truncation;
    __attribute__((aligned(16))) int SearchEdgeWidth = searchSize / 2;//Use integer truncation.

    //Compute the number of vertical and horizontal shifts the filtering window
    //makes within the search area:
    __attribute__((aligned(16))) int NumberOfHSearchShifts = searchSize - (EdgeWidth * 2);
    __attribute__((aligned(16))) int NumberOfVSearchShifts = NumberOfHSearchShifts;

    //Declare variables for the search area limits:
    __attribute__((aligned(16))) int Left, Top;

    //Some SSE filterData type vectors we'll need:
    __m128 filterVec, backgroundVec, gaussianVec, result1, result2,
            nanWindowVec, snrWindowVec, gaussianSum;


    __attribute__((aligned(16))) float OriginalNoisyValue;

    //Store the weighting type for use later:
    __attribute__((aligned(16))) int WeightingType = weightingFunctionType;

    //Store the linear fit bool:
    bool UsingLinearFitOption = false;
    if (useLinearFit > 0) {
        UsingLinearFitOption = true;
    }

    __attribute__((aligned(16))) int WeightNumberThreshold = minNumberOfWeights;

    //Center of Gaussian array:
    __attribute__((aligned(16))) int GaussianCenterOffset = ((windowSize * windowSize) - 1) / 2;

    //Grab a scanline for processing:
    #pragma omp critical
    {
        CurrentScanline = synchronizationCounter;
        synchronizationCounter++;
    }

    //	unsigned char* NaNMask = NaNMask;
    //unsigned char* SNRMask = SNRMask;
    float *UnfilteredData = iData;
    float *FilteredData = oData;
    float *SNRData = SNR;

    //Loop over filterData and filter:
    while (CurrentScanline < TotalScanlines)
    {
        //Set the coordinate of the center of the filtering window
        //based on the scanline. A scanline of value 0 starts at
        //the row equal to the edge width in the original filterData array:
        int Row = CurrentScanline;

        //Loop over the scanline, extracting the filter windows
        //and sliding them across the background search area, filtering:

        for (int Column = 0; Column < NumberOfHorizontalShifts; Column++) {
            //Check to see if the current value is NaN, and if
            //the current value has a SNR of 0.0. If not, filter.
            //If so..bypass:
            unsigned char NaNMaskValue = *(NaNMask + ((Row + EdgeWidth) * dataWidth) + (Column + EdgeWidth));
            unsigned char SNRMaskValue = *(SNRMask + ((Row + EdgeWidth) * dataWidth) + (Column + EdgeWidth));
            float SNRDataValue = *(SNRData + ((Row + EdgeWidth) * dataWidth) + (Column + EdgeWidth));

            if ((NaNMaskValue > 0) && (SNRMaskValue > 0) && (SNRDataValue > 0.0f)) {

                //Extract the filtering window and it's associated NaN mask
                // and SNR window based on the row and column centers.
                int increment = 0;
                for (int row = 0; row < windowSize; row++) {
                    for (int column = 0; column < windowSize; column++) {
                        *(SlidingWindow + increment) = *(UnfilteredData + ((Row + row) * dataWidth) +
                                                         (Column + column));
                        *(NaNMaskWindow + increment) = (float) (*(NaNMask + ((Row + row) * dataWidth) +
                                                                  (Column + column)));
                        *(SNRSlidingWindow + increment) = *(SNRData + ((Row + row) * dataWidth) + (Column + column));
                        increment++;
                    }
                }

                //Check to see if user has selected "adaptive H" option:
                if (useAdaptiveH > 0) {
                    //They are. Reset the adaptive variables:
                    _mm_store_ss(&Zero, N);
                    _mm_store_ss(&Zero, Mean);
                    _mm_store_ss(&Zero, StandardDeviation);

                    //Compute the total number of non-NaN values in the current
                    //filtering window by summing over the NaN mask (in the mask, NaN
                    //values are equal to 0, cast to float in NaNMaskWindow):
                    for (int x = 0; x < (windowSize * windowSize); x++) {
                        if (*(NaNMaskWindow + x) > 0.0f) {
                            N = _mm_add_ss(N, One);
                            Mean = _mm_add_ss(Mean, _mm_load_ss(SlidingWindow + x));
                        }
                    }

                    //Compute window mean:
                    Mean = _mm_div_ss(Mean, _mm_mul_ss(N, N));

                    //Loop through and compute the standard deviation,
                    //again checking for NaNs using the NaN mask:
                    for (int x = 0; x < (windowSize * windowSize); x++) {
                        if (*(NaNMaskWindow + x) > 0.0f) {
                            __m128 value = _mm_load_ss(SlidingWindow + x);
                            StandardDeviation =
                                    _mm_add_ss(StandardDeviation,
                                               _mm_mul_ss(_mm_sub_ss(Mean, value), _mm_sub_ss(Mean, value)));
                        }
                    }

                    //Final StdDev value for this sliding window:
                    StandardDeviation = _mm_mul_ss(StandardDeviation,
                                                   _mm_div_ss(One, _mm_sub_ss(_mm_mul_ss(N, N), One)));

                    //Recompute h, hSquared, and coefficient:
                    h = _mm_mul_ss(hOriginal, _mm_sqrt_ss(StandardDeviation));
                    hSquared = _mm_mul_ss(h, h);
                    coefficient =
                            _mm_div_ss(_mm_load_ss(&NegOne), _mm_mul_ss(_mm_load_ss(&Two), hSquared));

                }//End of "if UsingAdaptiveH" block.

                //Reset the running sum variables to zero:
                WeightValueSum = 0.0f;
                ScalarValue3Sum = 0.0f;

                //Get the search area limits based on the current filtering pixel.
                //(In hindsight I only needed the left and top values):
                GetSearchLimits(Column + EdgeWidth, Row + EdgeWidth, searchSize, dataWidth,
                                dataHeight, &Left, &Top);

                //Initialize the weight counter and threshold:
                int NumberOfWeightsAboveThreshold = 0;


                //Loop over search area, sliding window and filtering:
                for (int VPosition = 0; VPosition < NumberOfVSearchShifts; VPosition++) {
                    for (int HPosition = 0; HPosition < NumberOfHSearchShifts; HPosition++) {

                        //Check for the presence of a NaN at the center of the
                        //background search patch. Bypass filtering operation
                        //if NaN is present:
                        float CenterValue =
                                (float) (*(NaNMask + ((Top + VPosition + EdgeWidth) * dataWidth) +
                                           (Left + HPosition + EdgeWidth)));

                        if (CenterValue > 0.0f) {
                            //Extract background window, SNR window and NaN mask into array (NaNMaskWindow
                            //was extracted earlier):
                            increment = 0;
                            for (int row = 0; row < windowSize; row++) {
                                for (int column = 0; column < windowSize; column++) {
                                    *(BackgroundWindow + increment) =
                                            *(UnfilteredData + ((Top + VPosition + row) * dataWidth) +
                                              (Left + HPosition + column));

                                    *(NaNMaskBackground + increment) =
                                            (float) (*(NaNMask + ((Top + VPosition + row) * dataWidth) +
                                                       (Left + HPosition + column)));

                                    *(SNRBackgroundWindow + increment) =
                                            *(SNRData + ((Top + VPosition + row) * dataWidth) +
                                              (Left + HPosition + column));

                                    increment++;
                                }
                            }

                            //Normalize the Gaussian kernel to the sum of all the gaussian
                            //values in the kernel that do not correspond to a NaN value in
                            //either the background NaN mask or the window NaN mask. Then,
                            //multiply by the SNR window. If the user is not using SNR
                            //filterData, then the SNR window has been filled with 1.0 earlier.
                            //Use SSE (exclude SSE3 summing across register instruction):
                            float GaussianNormalizationTerm = 0.0f;
                            for (int x = 0; x < NumberOfWindowElements; x += 4) {
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
                                        _mm_mul_ps(_mm_load_ps(GaussianWindow + x),
                                                   _mm_mul_ps(nanWindowVec, snrWindowVec));

                                //Store this result to the gaussian buffer for use later, and
                                //extract and sum (no SSE3) to normalize:
                                _mm_store_ps(Register, gaussianVec);//for accumulation
                                _mm_store_ps((NaNSNRNormalizedGaussian + x), gaussianVec);//For final normalization.

                                //Accumulate:
                                GaussianNormalizationTerm +=
                                        (*(Register) + *(Register + 1) + *(Register + 2) + *(Register + 3));
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
                                float AccumulatedGaussian = 0.0f;
                                for (int a = 0; a < windowSize; a++) {
                                    for (int b = 0; b < windowSize; b++) {
                                        AccumulatedGaussian += *(NaNSNRNormalizedGaussian + increment);
                                        increment++;
                                    }
                                }

                                if (AccumulatedGaussian == 0.0f) {
                                    //Reset the gaussian center point to 1.0f. That means
                                    //that for this particular filterData point in the search area
                                    //it is as if the user has opted to include the center point
                                    //value. This avoids the situation where you have a gaussian
                                    //matrix consisting only of zeros:
                                    *(NaNSNRNormalizedGaussian + GaussianCenterOffset) = 1.0f;

                                    //Change the normalization term to 1.0f to reflect the change:
                                    GaussianNormalizationTerm = 1.0f;
                                }
                            }

                            //Loop through the gaussian buffer and normalize with the above
                            //term (don't bother checking for zeros, just compute):
                            gaussianSum = _mm_load1_ps(&GaussianNormalizationTerm);//Populate vector
                            for (int x = 0; x < NumberOfWindowElements; x += 4) {
                                //Final normalization:
                                gaussianVec =
                                        _mm_div_ps(_mm_load_ps(NaNSNRNormalizedGaussian + x), gaussianSum);


                                _mm_store_ps((NaNSNRNormalizedGaussian + x), gaussianVec);
                            }

                            //Initialize the iterations sum variable:
                            //IterationsSum = 0.0f;
                            IterationsSum = _mm_load_ss(&Zero);

                            //Store the original noisy value (at the center of the background window):
                            OriginalNoisyValue =
                                    *(UnfilteredData + ((Top + VPosition + EdgeWidth) * dataWidth) +
                                      (Left + HPosition + EdgeWidth));

                            //Loop over vectorized window and filter:
                            for (int x = 0; x < NumberOfWindowElements; x += 4) {
                                //Load filterData:
                                filterVec = _mm_load_ps(SlidingWindow + x);
                                backgroundVec = _mm_load_ps(BackgroundWindow + x);

                                //Load NaN-normalized / SNR-weighted gaussian:
                                gaussianVec = _mm_load_ps(NaNSNRNormalizedGaussian + x);

                                //compute difference between window and background:
                                result1 = _mm_sub_ps(filterVec, backgroundVec);

                                //Square the difference and multiply by the gaussian:
                                result2 = _mm_mul_ps(gaussianVec, _mm_mul_ps(result1, result1));

                                //Sum result2 across registers (without using SSE3).
                                //First, store in buffer:
                                _mm_store_ps(Register, result2);
                                Temp = 0.0f;

                                //Accumulate:
                                Temp +=
                                        (*(Register) + *(Register + 1) + *(Register + 2) + *(Register + 3));

                                //Add iteration sum to running sum:
                                //IterationsSum += Temp;
                                IterationsSum = _mm_add_ss(IterationsSum, _mm_load_ss(&Temp));

                            }//End of window element loop.

                            //Next step is to compute the weight value according to
                            //the user selection:

                            if (WeightingType == 1)//Standard
                            {
                                float temp;
                                _mm_store_ss(&temp, _mm_mul_ss(coefficient, IterationsSum));
                                WeightValue = expf(temp);
                            } else if (WeightingType == 2)//Bisquare
                            {
                                //float r = sqrt(IterationsSum);
                                float r;
                                _mm_store_ss(&r, _mm_sqrt_ss(IterationsSum));

                                //compare against the noise parameter value "h":
                                float hCompare;
                                _mm_store_ss(&hCompare, h);

                                if (r <= hCompare) {
                                    //Compute weight via Bisquare method:
                                    __m128 Term = _mm_sub_ss(One, _mm_div_ss(IterationsSum, hSquared));
                                    _mm_store_ss(&WeightValue, _mm_mul_ss(Term, Term));
                                } else {
                                    //Set to zero:
                                    WeightValue = 0.0f;
                                }
                            } else //Modified Bisquare
                            {
                                //float r = sqrt(IterationsSum);
                                float r;
                                _mm_store_ss(&r, _mm_sqrt_ss(IterationsSum));

                                //compare against the noise parameter value "h":
                                float hCompare;
                                _mm_store_ss(&hCompare, h);

                                if (r <= hCompare) {
                                    //Compute weight via Modified Bisquare method:
                                    float Term;
                                    _mm_store_ss(&Term, _mm_sub_ss(One, _mm_div_ss(IterationsSum, hSquared)));
                                    WeightValue = pow(Term, 8);
                                } else {
                                    //Set to zero:
                                    WeightValue = 0.0f;
                                }
                            }

                            //Check for linear fit option:
                            if (UsingLinearFitOption)
                            {

                            }//end of "if UsingLinearFitOption" block.
                            else {
                                //Store to WeightSum:
                                WeightValueSum += WeightValue;

                                //Compute the ScalarValue3 running total for this filtering position:
                                ScalarValue3Sum += (WeightValue * OriginalNoisyValue);
                            }//end of "else averaging" block.

                        }//End of "if CenterValue > 0.0f" block.

                    }//End of "for(int HPosition = 0; ... "
                }//End of "for(int VPosition = 0; ..."


                //check for linear regression fit option:
                if (UsingLinearFitOption)
                {
                    std::cout<< "Linear Regression fit option is not supported !!"<<std::endl;

                }//End of "if UsingLinearFit" block.
                else {
                    //Compute and store final result:
                    *(FilteredData + ((Row + EdgeWidth) * dataWidth) + (Column + EdgeWidth)) =
                            (float) (ScalarValue3Sum / WeightValueSum);
                }//End of averaging lock.

            }//End of "if (NanMask[][] == 1.0f) && SNR > 0.0f " block.

            else {
                //Store a NaN at this location since there is either
                //a NaN in the original filterData or the SNR is too low:
                *(FilteredData + ((Row + EdgeWidth) * dataWidth) +
                  (Column + EdgeWidth)) = std::numeric_limits<float>::quiet_NaN();
            }

        }//End of scanline loop (int Column = 0;...).

        //Get the next scanline for processing:
        #pragma omp critical
        {
            CurrentScanline = synchronizationCounter;
            synchronizationCounter++;
        }

    }//End of filtering loop (while CurrentScanline <= ...).

    //Deallocate memory here:
    _mm_free(SlidingWindow);
    _mm_free(NaNMaskWindow);
    _mm_free(BackgroundWindow);
    _mm_free(GaussianWindow);
    _mm_free(NaNSNRNormalizedGaussian);
    _mm_free(NaNMaskBackground);
    _mm_free(Register);
    _mm_free(SNRBackgroundWindow);
    _mm_free(SNRSlidingWindow);
}



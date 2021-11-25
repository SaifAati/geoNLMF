#pragma once

class CNLMFilter
{

    public:
    int nbCPUs; // Number of system CPUs.

    // Thread synchronization variable:
    unsigned int synchronizationCounter;

    int iDataType;
    int windowSize;


    // Constructors:
	CNLMFilter();

	void SetDataWidth(int width);
	void SetDataHeight(int height);
	void SetSearchSize(int size);
	void SetWindowSize(int size);
	void StoreWeightingFunctionType(int type);
	void StoreMinNumberOfWeights(int number);
	void StoreMinimumWeightValue(double value);	
	void StoreLinearFitOption(int selection);
	void StoreAdaptiveHoption(int selection);
	void StoreSNRWeightingOption(int selection);
	void StoreCenterPointOption(int selection);
	void StoreHparam(double h);

	void StoreInputArrayPtr(float* inputPtr);
	void StoreOutputArrayPtr(float* outputPtr);

	void StoreSNRArrayPtr(float* snrPtr);
	void ComputeOutputDimensions();
	void AllocateNaNMask();
	void AllocateSNRMask();
	void ComputeGaussianWindow();


	double GetMinimumWeightThreshold();
	int GetDataWidth();
	int GetDataHeight();
	int GetWindowSize();
	int GetOutputHeight();
	int GetOutputWidth();
	float* GetInputArrayPtr();
	float* GetOutputArrayPtr();
	unsigned char*  GetNaNMask();

	bool UsesSNRdata();

    void GenerateNaNMask();  // Threaded.
    void ReplaceNaNValsInSNRArray();  // Threaded.


    void Filter_SNR();  // Threaded.
    void Filter();      // Threaded.

    void RestoreNaNValsInSNRArray();  // Threaded.

    void PlaceNaNValsInOutputArray(); // Threaded.


    private:

	int     dataWidth;             // The number of columns of the input array.
	int     dataHeight;            // The number of rows of the input array.
	int     oWidth;           // The size of the output array width minus the non-filtered edges.
	int     oHeight;          // The size of the output array height minus the non-filtered edges.
	int     searchSize;            // The square dimension of the search area.
	int     weightingFunctionType; // 1 = Standard, 2 = Bisquare, 3 = Modified Bisquare.
	int     centerPointInclusive;  // Include (1) or omit (0) filter point in calculation.
	int     useLinearFit;          // Linear fit of weight data (1) vs. simple averaging (0).
	int     useAdaptiveH;          // Adaptive H (1), or original algorith H (0).
	int     useSNRdata;            // SNR weighting, yes = 1, no = 0.
	double  minimumWeightThreshold;// Minimum value necessary for inclusion.
	int     minNumberOfWeights;    // Minimum number of weights to apply linear fit.
	double  hParam;				   // The H (noise) estimation parameter.
	float*  iData;           // Pointer to Float-type input array.
	float*  SNR;                 // Pointer to Float-type SNR array.
	float*  oData;          // Pointer to Float-type output array.
	float*  gaussianWindow;      // Pointer to Float-type gaussian window.
	int     CurrentScanline;       // Used by worker threads to synchronize processing.
	unsigned char*  SNRMask;       // Used to store the overall SNR mask.
	unsigned char*  NaNMask;       // Used to store the overall NaNMask;
	void GetSearchLimits(int XPos, int YPos, int SearchDimension, int dataWidth,
		  			     int dataHeight, int* left, int* top);

};
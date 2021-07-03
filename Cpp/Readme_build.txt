## g++ -fpic CStatisticalCorrelator.cpp -I include/ `gdal-config --cflags` `gdal-config --libs` -fopenmp

##source /opt/intel/oneapi/setvars.sh --force

## icpx -c -fpic StaticSincResampler.cpp -fopenmp


"uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,


  g++ -fpic -c -Wall mainStatCorrlib.cpp CStatisticalCorrelator.cpp -fopenmp -I Include/

  g++ -fpic -Wall -o libcstatcorr.so *.o -shared -fopenmp -I Include/
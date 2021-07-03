g++ -fpic -c  geoNLMFLib.cpp geoNLMF.cpp -I Include/ -fopenmp
g++ -fpic -o libgeoNLMF.so *.o -shared -I Include/ -fopenmp

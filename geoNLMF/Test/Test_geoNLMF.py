import os
from geoNLMF.geoNLMF import cgeoNLMF

################################# EXAMPLE 1 ############################################################################
iRasterPath = os.path.join(os.path.dirname(__file__), "Data/Madoi/T47SMU_20201012T040719_B02_VS_T47SMU_20211017T040749_B02_geoFqCorr_W128_S8_det.tif")
geoNLMFObj = cgeoNLMF(iRasterPath=iRasterPath,
                      patchSize=7,
                      searchSize=41,
                      h=1.5,
                      useSNR=0,
                      adaptive=0,
                      visualize=True,
                      bandsList=[1],
                      factor=0.75,
                      debug=True)
geoNLMFObj.geoNLMF()
del geoNLMFObj



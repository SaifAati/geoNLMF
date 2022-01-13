import os
from geoNLMF.geoNLMF import cgeoNLMF

iRasterPath = os.path.join(os.path.dirname(__file__), "Data/Disp2.tif")
# iRasterPath = "/home/cosicorr/0-WorkSpace/geoCosiCorr3D_TestWorksapce/RFM_Ortho_Dev/TestRFM_Refinement/20201015_033837_0e3a_1B_Analytic_DN_oRFM_4_noDEM_VS_20210921_034958_1005_1B_Analytic_DN_oRFM_4_noDEM_geoFqCorr_W64_S8.tif"
geoNLMFObj = cgeoNLMF(iRasterPath=iRasterPath,
                      patchSize=5,
                      searchSize=41,
                      h=1.5,
                      useSNR=0,
                      adaptive=0,
                      visualize=True,
                      bandsList=[1])
geoNLMFObj.geoNLMF()


# import numpy as np
# print(list(np.arange(1,10,1)))
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import geospatialroutine.georoutines as geoRT
from ctypes import cdll


class cgeoNLMF:
    __iRasterInfo = None

    __oArray = None

    __height = None
    __width = None

    __useSNR = 0
    __patchSize = 5
    __searchSize = 21
    __h = 1.5

    __adaptive = 0
    __centerPoint = 0
    __weighting = 2
    __linearRegression = 0
    __minWeight = 0.1
    __minNumberWeights = 6
    __nbBands = 1
    __snrArray_fl = np.zeros((1, 1), dtype=np.float32)
    _oRasterPath = ""
    __geoNLMFLibPath = "/home/cosicorr/0-WorkSpace/3-PycharmProjects/geoNLMF_Dev/geoNLMF_v0.1/Cpp/libgeoNLMF.so"

    def __init__(self, iRasterPath,
                 useSNR=0,
                 patchSize=None,
                 searchSize=None,
                 h=None,
                 adaptive=None,
                 centerPoint=None,
                 weighting=None,
                 linearRegression=None,
                 minWeight=None,
                 minNumberWeights=None,
                 nbBands=None,
                 oRasterPath=None

                 ):
        """

        Args:
            iRasterPath:
            useSNR:
            patchSize:
            searchSize:
            h:
            adaptive:
            centerPoint:
            weighting:
            linearRegression:
            minWeight:
            minNumberWeights:
            nbBands:
            oRasterPath:
        """
        self.iRasterPath = iRasterPath
        if useSNR != None:
            self.__useSNR = useSNR
        if patchSize != None:
            self.__patchSize = patchSize
        if searchSize != None:
            self.__searchSize = 21
        if h != None:
            self.__h = h
        if adaptive != None:
            self.__adaptive = adaptive
        if centerPoint != None:
            self.__centerPoint = 0
        if weighting != None:
            self.__weighting = weighting
        if linearRegression != None:
            self.__linearRegression = linearRegression
        if minWeight != None:
            self.__minWeight = minWeight
        if minNumberWeights != None:
            self.__minNumberWeights = minNumberWeights
        if nbBands != None:
            self.__nbBands = nbBands
        if oRasterPath != None:
            self._oRasterPath = os.path.join(os.path.dirname(self.iRasterPath),
                                             Path(self.iRasterPath).stem + "_geoNLMF.tif")

    def geoNLMF(self):

        self.__iRasterInfo = geoRT.RasterInfo(self.iRasterPath)
        self.iRasterArray = self.__iRasterInfo.ImageAsArray(bandNumber=self.__nbBands)
        self.__height = self.__iRasterInfo.rasterHeight
        self.__width = self.__iRasterInfo.rasterWidth

        if self.__useSNR == 1:
            snrArray = self.__iRasterInfo.ImageAsArray(bandNumber=3)
            self.__snrArray_fl = np.array(snrArray.flatten(), dtype=np.float32)
        self.PerformNLMF()
        self.VisualizeFiltering(-2, 2)

        return

    def VisualizeFiltering(self, vmin=-0.1, vmax=0.1, cmap="RdYlBu"):
        """

        Args:
            vmin:
            vmax:
            cmap:

        Returns:

        """
        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(self.iRasterArray, cmap=cmap, vmin=vmin, vmax=vmax)
        # oArray = np.empty(np.shape(self.__oArray))
        # oArray[:] = float(self.__oArray[:])
        axs[1].imshow(self.__oArray, cmap=cmap, vmin=vmin, vmax=vmax)
        # axs[2].imshow(self.snrOutput, cmap="gray", vmin=0, vmax=1)
        for ax, title in zip(axs, ["Before geoNLMF", "after geoNLMF"]):
            ax.axis('off')
            ax.set_title(title)
        # plt.savefig(os.path.join(os.path.dirname(self.oCorrPath), Path(self.oCorrPath).stem + ".png"), dpi=600)
        plt.show()
        return

    def PerformNLMF(self):
        """
        
        Returns:

        """
        # load the library
        libCstatCorr = cdll.LoadLibrary(self.__geoNLMFLibPath)
        libCstatCorr.InputData.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32)]

        patchSize = np.array([self.__patchSize], dtype=np.int32)
        searchAreaSize = np.array([self.__searchSize], dtype=np.int32)
        h = np.array([self.__h], dtype=np.float32)
        weightingMethod = np.array([self.__weighting], dtype=np.int32)

        minWeightValue = np.array([self.__minWeight], dtype=np.float32)
        minNumberOfWeights = np.array([self.__minNumberWeights], dtype=np.int32)
        linear = np.array([self.__linearRegression], dtype=np.int32)
        adaptive = np.array([self.__adaptive], dtype=np.int32)
        SNRint = np.array([self.__useSNR], dtype=np.int32)
        CenterInclusive = np.array([self.__centerPoint], dtype=np.int32)
        dimension = np.array([self.__height, self.__width], dtype=np.int32)
        iArray_fl = np.array(self.iRasterArray.flatten(), dtype=np.float32)
        oArray_fl = np.zeros((self.__width * self.__height, 1), dtype=np.float32)
        snrArray_fl = np.copy(self.__snrArray_fl)

        libCstatCorr.InputData(patchSize,
                               searchAreaSize,
                               h,
                               weightingMethod,
                               minWeightValue,
                               minNumberOfWeights,
                               linear,
                               adaptive,
                               SNRint,
                               CenterInclusive,
                               dimension,
                               iArray_fl,
                               oArray_fl,
                               snrArray_fl)

        self.__oArray = np.asarray(oArray_fl[:, 0]).reshape((self.__height, self.__width))

        return


if __name__ == '__main__':
    iRasterPath = os.path.join(os.path.dirname(__file__), "Test/Data/pyCorr_Stat_B1_R20_subset.tif")
    print(iRasterPath)
    geoNLMFObj = cgeoNLMF(iRasterPath, patchSize=3, searchSize=42, h=2, useSNR=0,adaptive=1,

                 weighting=1)
                 # linearRegression=None,
                 # minWeight=None,
                 # minNumberWeights=None,
                 # nbBands=None,
                 # oRasterPath=None)
    geoNLMFObj.geoNLMF()

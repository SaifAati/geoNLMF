import os, sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import geoRoutines.georoutines as geoRT
from ctypes import cdll
from tqdm import tqdm


class cgeoNLMF:
    __useSNR = 0
    __patchSize = 5
    __searchSize = 21
    __h = 1.5
    __centerPoint = 0
    __weighting = 2
    __linearRegression = 0
    __minWeight = 0.1
    __minNumberWeights = 6
    __snrArray_fl = np.zeros((1, 1), dtype=np.float32)

    __geoNLMFLibPath = os.path.join(os.path.dirname(__file__), "lib/build/src/libgeoNLMF.v0.0.3.so")
    libgeoNLMF = cdll.LoadLibrary(__geoNLMFLibPath)

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
                 bandsList=[],
                 oRasterPath="",
                 visualize=False,
                 debug=False,
                 factor = 0.75
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
        self.debug = debug
        self.iRasterPath = iRasterPath
        self.factor = factor
        self.searchSize = searchSize
        if self.searchSize == None:
            self.searchSize = self.__searchSize

        self.useSNR = useSNR
        if self.useSNR == None:
            self.useSNR = self.__useSNR
        self.patchSize = patchSize
        if self.patchSize == None:
            self.patchSize = self.__patchSize

        self.h = h
        if self.h == None:
            self.h = self.__h
        self.adaptive = adaptive

        if self.adaptive != None:
            self.adaptive = adaptive

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
        self.bandsList = bandsList
        self.oRasterPath = oRasterPath
        if oRasterPath == "":
            self.oRasterPath = os.path.join(os.path.dirname(self.iRasterPath),
                                            Path(self.iRasterPath).stem + "_geoNLMF.tif")
        self.visualize = visualize

    def geoNLMF(self):

        self.iRasterInfo = geoRT.RasterInfo(self.iRasterPath)
        self.height = self.iRasterInfo.rasterHeight
        self.width = self.iRasterInfo.rasterWidth

        if self.useSNR == 1:
            snrArray = self.iRasterInfo.ImageAsArray(bandNumber=3)
            self.snrArray_fl = np.array(snrArray.flatten(), dtype=np.float32)
        else:
            self.snrArray_fl = np.copy(self.__snrArray_fl)

        if len(self.bandsList) == 0:
            self.bandsList = list(np.arange(1, self.iRasterInfo.nbBand + 1, 1))

        oArrayList = []
        for band_ in tqdm(self.bandsList, "Perfroming geoNLMF"):
            print("______BAND:", band_, "____________")

            self.iRasterArray = self.iRasterInfo.ImageAsArray(bandNumber=int(band_))
            self.oArray = self.PerformNLMF()
            oArrayList.append(self.oArray)


        if self.debug:
            print("oRasterPath:", self.oRasterPath)
            print("ogeoTrans:", self.iRasterInfo.geoTrans)

        geoRT.WriteRaster(oRasterPath=self.oRasterPath,
                          geoTransform=self.iRasterInfo.geoTrans,
                          arrayList=oArrayList,
                          epsg=self.iRasterInfo.EPSG_Code,
                          noData=-32767)
        if self.visualize:
            self.VisualizeFiltering(self.iRasterInfo,geoRT.RasterInfo(self.oRasterPath))
        return

    def PerformNLMF(self):
        """
        
        Returns:

        """
        # load the library

        self.libgeoNLMF.InputData.argtypes = []
        self.libgeoNLMF.InputData.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
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

        patchSize = np.array([self.patchSize], dtype=np.int32)
        searchAreaSize = np.array([self.searchSize], dtype=np.int32)
        print(searchAreaSize)
        h = np.array([self.h], dtype=np.float32)
        weightingMethod = np.array([self.__weighting], dtype=np.int32)

        minWeightValue = np.array([self.__minWeight], dtype=np.float32)
        minNumberOfWeights = np.array([self.__minNumberWeights], dtype=np.int32)
        linear = np.array([self.__linearRegression], dtype=np.int32)
        adaptive = np.array([self.adaptive], dtype=np.int32)
        SNRint = np.array([self.useSNR], dtype=np.int32)
        CenterInclusive = np.array([self.__centerPoint], dtype=np.int32)
        dimension = np.array([self.height, self.width], dtype=np.int32)
        iArray_fl = np.array(self.iRasterArray.flatten(), dtype=np.float32)
        iArray_fl = np.ma.masked_invalid(iArray_fl)

        oArray_fl = np.zeros((self.width * self.height, 1), dtype=np.float32)
        snrArray_fl = np.copy(self.snrArray_fl)
        print(iArray_fl)
        self.libgeoNLMF.InputData(patchSize,
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

        oArray = np.asarray(oArray_fl[:, 0]).reshape((self.height, self.width))
        # print(self.oArray)
        self.libgeoNLMF.InputData.argtypes = []
        return oArray

    def VisualizeFiltering(self, inRasterInfo,oRasterInfo, vmin=None, vmax=None, cmap="RdYlBu", save=True):
        """

        Args:
            vmin:
            vmax:
            cmap:

        Returns:

        """

        from geoRoutines.Plotting.Plotting_Routine import ColorBar_

        bands = oRasterInfo.nbBand
        for band in range(bands):
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))
            inArray = inRasterInfo.ImageAsArray(bandNumber=band+1)
            filteredArray = oRasterInfo.ImageAsArray(bandNumber=band+1)

            stat = geoRT.cgeoStat(inputArray=inArray, displayValue=False)

            factor = self.factor
            if vmin == None:
                vmin = float(stat.mean) - factor * float(stat.std)
            if vmax == None:
                vmax = float(stat.mean) + factor * float(stat.std)

            im1 = axs[0].imshow(inArray, cmap=cmap, vmin=vmin, vmax=vmax)
            im2 = axs[1].imshow(filteredArray, cmap=cmap, vmin=vmin, vmax=vmax)
            ColorBar_(ax=axs[0], mapobj=im1, cmap=cmap, vmin=vmin, vmax=vmax, orientation="vertical")
            ColorBar_(ax=axs[1], mapobj=im2, cmap=cmap, vmin=vmin, vmax=vmax, orientation="vertical")
            for ax, title in zip(axs, ["Disp. before\ngeoNLMF  ", "Disp. after \ngeoNLMF"]):
                ax.axis('off')
                ax.set_title(title)

            if save:
                plt.savefig(os.path.join(os.path.dirname(self.oRasterPath),
                                         Path(self.oRasterPath).stem + "_b" + str(band+1) + ".png"), dpi=300)
            fig.clear()

        return




import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import geoRoutines.georoutines as geoRT
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

    __geoNLMFLibPath = os.path.join(os.path.dirname(__file__), "lib/build/src/libgeoNLMF.so")

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
                 oRasterPath="",
                 visualize=False

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
        if oRasterPath == "":
            self.oRasterPath = os.path.join(os.path.dirname(self.iRasterPath),
                                            Path(self.iRasterPath).stem + "_B" + str(self.__nbBands) + "_geoNLMF.tif")
        self.visualize = visualize

    def geoNLMF(self):

        self.__iRasterInfo = geoRT.RasterInfo(self.iRasterPath)
        # TODO: noData values are not supported
        ## read noData value then --> nan
        self.iRasterArray = self.__iRasterInfo.ImageAsArray(bandNumber=self.__nbBands)

        self.__height = self.__iRasterInfo.rasterHeight
        self.__width = self.__iRasterInfo.rasterWidth

        if self.__useSNR == 1:
            snrArray = self.__iRasterInfo.ImageAsArray(bandNumber=3)
            self.__snrArray_fl = np.array(snrArray.flatten(), dtype=np.float32)
        self.PerformNLMF()
        print(self.oRasterPath)
        print(self.__iRasterInfo.geoTrans)
        geoRT.WriteRaster(oRasterPath=self.oRasterPath, geoTransform=self.__iRasterInfo.geoTrans,
                          arrayList=[self.__oArray], epsg=self.__iRasterInfo.EPSG_Code)
        if self.visualize:
            self.VisualizeFiltering(-2.5, 2.5)

        return

    def VisualizeFiltering(self, vmin=-0.1, vmax=0.1, cmap="RdYlBu"):
        """

        Args:
            vmin:
            vmax:
            cmap:

        Returns:

        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axs = plt.subplots(1, 2)

        im = axs[0].imshow(self.iRasterArray, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].imshow(self.__oArray, cmap=cmap, vmin=vmin, vmax=vmax)
        for ax, title in zip(axs, ["E/W Disp. before\ngeoNLMF  ", "E/W Disp. after \ngeoNLMF"]):
            ax.axis('off')
            ax.set_title(title)
        # divider = make_axes_locatable(axs[-1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(os.path.dirname(self.oRasterPath), Path(self.oRasterPath).stem + ".png"), dpi=300)
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


def PlotProfiles():
    ## plot profile:
    import geoRoutines.Plotting.Plot_Profiles as pltProfileRT

    profilePath = os.path.join(os.path.dirname(__file__), "Test/Data/Profile_pline.kml")

    rasterPath1 = os.path.join(os.path.dirname(__file__), "Test/Data/Disp2.tif")
    rasterPath2 = os.path.join(os.path.dirname(__file__), "Test/Data/Disp2_B1_geoNLMF.tif")
    # profile = pltProfileRT.ExtractProfile(rasterPath=rasterPath,
    #                                       profilePath=profilePath, bandNumber=1, offset=True, center=2700,
    #                                       center_minus=100, center_plus=100)
    profile1 = pltProfileRT.ExtractProfile(rasterPath=rasterPath1,
                                           profilePath=profilePath, bandNumber=1, offset=False)
    profile2 = pltProfileRT.ExtractProfile(rasterPath=rasterPath2,
                                           profilePath=profilePath, bandNumber=1, offset=False)
    labels = ["orig.", "withgeoNLMF"]
    fig1, (ax1) = plt.subplots(1, 1)
    for index, profileObj_ in enumerate([profile1, profile2]):
        ax1.tick_params(direction='in', top=True, right=True, which="both", axis="both", labelsize=14)
        ax1.plot(profileObj_.cumDistances(), profileObj_.profileValues, linewidth="3",
                 label=labels[index])
    ax1.grid()
    ax1.minorticks_on()
    ax1.set_xlabel("Distance along profile [m]", fontsize=14)
    ax1.set_ylabel("Displacement [m]", fontsize=14)
    # ax1.set_ylim(-2.5, 1)
    # ax1.set_xlim(self.cumDistances()[0], self.cumDistances()[-1])
    plt.legend()
    plt.show()
    profile1.PlotProfile()

    return


if __name__ == '__main__':
    iRasterPath = os.path.join(os.path.dirname(__file__), "Test/Data/Disp2.tif")
    geoNLMFObj = cgeoNLMF(iRasterPath=iRasterPath, patchSize=7, searchSize=41, h=2, useSNR=0, adaptive=0,
                          visualize=True, nbBands=1)
    geoNLMFObj.geoNLMF()

import numpy as np
import copy

class HorizontalBase(object):
    def __init__(self, model_name, samplesStatsMap, model_count=13):
        self.model_count = model_count
        self.samplesStatsMap = samplesStatsMap
        self.__name__ = model_name

    def normalizeZScore(self, X):
        assert(X[0].shape[0] == self.model_count)
        X_zscore = copy.deepcopy(X)
        for iLab in range(self.model_count):
            m = self.samplesStatsMap[iLab]['mean']
            s = self.samplesStatsMap[iLab]['std']
            for p in range(len(X)):
                for jTime in range(X[p].shape[1]):
                    X_zscore[p][iLab,jTime] = (X[p][iLab,jTime]-m)/s # z-score
        return X_zscore

    def prepareXY(self, X, X_int, Y, labIndex, remove_y_nan=True):
        X = X.copy()
        X_int = X_int.copy()
        Y = Y.copy()
        yCol = Y[labIndex, :]
        if remove_y_nan:
            validIndexes = ~np.isnan(yCol)
            X = X[:, validIndexes]
            X_int = X_int[:, validIndexes]
            yCol = yCol[validIndexes]
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.isnan(X[i, j]):
                    X[i, j] = X_int[i, j]

        X_prepared = np.vstack([X[:labIndex, :], X_int[labIndex, :], X[(labIndex + 1):, :]])
        return X_prepared.transpose(), yCol.transpose()

    def splitIntoParticipants(self, arrFlat, binWidth):
        cumulativeSplit = np.cumsum([0] + binWidth)
        arr = []
        for i in range(len(cumulativeSplit) - 1):
            fromI = cumulativeSplit[i]
            toI = cumulativeSplit[i + 1]
            pY = arrFlat[fromI:toI, :].transpose()
            arr.append(pY)
        return arr

    def fill_nans(self, destArr, srcArr, mask=None):
        assert (len(destArr) == len(srcArr))
        for p in range(len(destArr)):
            for iLab in range(destArr[p].shape[0]):
                for jTime in range(destArr[p].shape[1]):
                    if np.isnan(destArr[p][iLab, jTime]):
                        destArr[p][iLab, jTime] = srcArr[p][iLab, jTime]
import numpy as np
import scipy


class AnticipatoryLicks:
    """simple class for containing anticipatory licking data"""
    def __init__(self, baseline_rate, anti_rate, corridor):
        nan_rates = np.isnan(baseline_rate) + np.isnan(anti_rate)
        self.baseline = baseline_rate[np.logical_not(nan_rates)]
        self.anti_rate = anti_rate[np.logical_not(nan_rates)]

        self.m_base = np.mean(self.baseline)
        self.m_anti = np.mean(self.anti_rate)

        self.corridor = int(corridor)
        self.anti = False
        if self.m_anti > 0:
            self.test = scipy.stats.wilcoxon(self.baseline, self.anti_rate)
            if (self.test[1] < 0.01) and (self.m_base < self.m_anti):
                self.anti = True
        else:
            self.test = [np.nan, 1]
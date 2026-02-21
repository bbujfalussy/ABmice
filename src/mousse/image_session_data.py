import pathlib
import json
from xml.dom import minidom

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

from .utils import *
from .stages import Stage
from .corridors import *
from .ImShuffle import *
from .anticipatory_licks import AnticipatoryLicks


class ImagingSessionData:
    'Base structure for both imaging and behaviour data'

    def __init__(self, datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, trigger_voltage_path,
                    sessionID=np.nan, selected_laps=None, speed_threshold=5, randseed=123, electrophysiology_analysis=False, reward_zones=None,
                    spikes_tag='', data_folder='analysed_data',
                    action_log_file_path: pathlib.Path = None,
                    trigger_log_file_path: pathlib.Path = None,
                    F_all_path: pathlib.Path = None,
                    spikes_all_path: pathlib.Path = None,
                    is_cell_path: pathlib.Path = None,
                 ):
        self.datapath: pathlib.Path = datapath
        self.date_time = date_time
        self.name = name
        self.task = task
        self.suite2p_folder: pathlib.Path = suite2p_folder
        self.imaging_logfile_name: pathlib.Path = imaging_logfile_name
        self.multiplane = False

        self.stage = 0
        self.stages = []
        self.sessionID = sessionID
        self.randseed = randseed
        self.selected_laps = selected_laps
        self.speed_threshold = speed_threshold
        self.electrophysiology_analysis = electrophysiology_analysis
        self.minimum_Nlaps = 3
        self.substage_change_laps = [0]
        self.substage_change_time = [0]
        self.data_folder = data_folder
        self.stage: int = self.get_stage(action_log_file_path=action_log_file_path)  # reads the stage of the experiment from the log file

        self.F_all = np.load(F_all_path)  # npy array, N_ROI x N_frames, fluorescence traces of ROIs from suite2p
        self.spks_all = np.load(spikes_all_path)  # npy array, N_ROI x N_frames, spike events detected from suite2p
        self.iscell = np.load(is_cell_path)  # np array, N_ROI x 2, 1st col: binary classified as cell. 2nd P_cell?

        self.parent_dir: pathlib.Path = pathlib.Path(__file__).parent

        with open(self.parent_dir / (self.task + '_stages.json'), 'r') as stages_file:
            self.stage_collection = StageCollection.from_json(json.load(stages_file))

        with open(self.parent_dir / (self.task + '_corridors.json'), 'r') as corridors_file:
            self.corridor_collection = CorridorCollection.from_json(json.load(corridors_file))


        self.all_corridors = np.hstack([0, np.array(
            self.stage_collection.stages[self.stage].corridors)])  # we always add corridor 0 - that is the grey zone

        if (reward_zones is not None):
            corridors_changed = np.unique(reward_zones[:, 0])
            for i_corrid in corridors_changed:
                zones_i = np.flatnonzero(reward_zones[:, 0] == i_corrid)
                zones_start_corrid = reward_zones[zones_i, 1]
                zones_end_corrid = reward_zones[zones_i, 2]
                self.corridor_collection.corridors[int(i_corrid)].reward_zone_starts = zones_start_corrid
                self.corridor_collection.corridors[int(i_corrid)].reward_zone_ends = zones_end_corrid
                print('reward zones added manually')

        ## in certain tasks, the same corridor may appear multiple times in different substages
        ## Labview uses different indexes for corridors in different substages, therefore
        ## we need to keep this corridor in the list self.corridors for running self.get_lapdata()
        ## but we should remove the redundancy after the data is loaded

        self.last_zone_start = 0
        self.last_zone_end = 0
        for i_corridor in self.all_corridors:
            if (i_corridor > 0):
                if (max(self.corridor_collection.corridors[i_corridor].reward_zone_starts) > self.last_zone_start):
                    self.last_zone_start = max(self.corridor_collection.corridors[i_corridor].reward_zone_starts)
                if (max(self.corridor_collection.corridors[i_corridor].reward_zone_ends) > self.last_zone_end):
                    self.last_zone_end = max(self.corridor_collection.corridors[i_corridor].reward_zone_ends)

        self.speed_factor = 106.5 / 3500.0  ## constant to convert distance from pixel to cm
        self.corridor_length_roxel = (self.corridor_collection.corridors[self.all_corridors[1]].length - 1024.0) / (
                    7168.0 - 1024.0) * 3500
        self.corridor_length_cm = self.corridor_length_roxel * self.speed_factor  # cm
        self.N_pos_bins = int(np.round(self.corridor_length_roxel / 70))

        self.imstart_time = locate_imaging(
            trigger_log_file_path= trigger_log_file_path,
            trigger_voltage_path=trigger_voltage_path
        )

        ##################################################
        ## loading imaging data
        ##################################################

        self.even_odd_rate_calculated = False
        self.start_end_rate_calculated = False


        # self.stat_string = self.suite2p_folder / 'stat.npy'  # we may load these later if needed
        # self.ops_string = self.suite2p_folder / 'ops.npy'
        # print('suite2p data loaded')

        imtimes_success = self.LoadImaging_times(self.imstart_time)
        if (imtimes_success == False):
            return
        self.frame_period = np.median(np.diff(self.frame_times))
        self.frame_pos = np.zeros(len(self.frame_times))  # position and
        self.frame_laps = np.zeros(len(self.frame_times))  # lap number for the imaging frames, to be filled later
        print('suite2p time axis loaded')

        ## arrays containing only valid cells
        self.neuron_index = np.nonzero(self.iscell[:, 0])[0]
        self.F = self.F_all[self.neuron_index, :]
        self.raw_spks = self.spks_all[self.neuron_index, :]
        self.dF_F = np.copy(self.F)
        self.spks = np.copy(self.raw_spks)  # could be normalized in calc_dF_F: spks / F / SD(F)
        self.N_cells = self.F.shape[0]
        # self.cell_SDs = np.sqrt(np.var(self.dF_F, 1)) - we calculate this later
        self.cell_SDs = np.zeros(self.N_cells)  # a vector with the SD of the cells
        self.cell_SNR = np.zeros(self.N_cells)  # a vector with the signal to noise ratio of the cells (max F / SD)
        self.calc_dF_F()
        self.detect_events()

        ##################################################
        ## loading behavioral data
        ##################################################

        self.ImLaps = []  # list containing a special class for storing the imaging and behavioral data for single laps
        self.n_laps = 0  # total number of laps
        self.i_Laps_ImData = np.zeros(1)  # np array with the index of laps with imaging
        self.i_corridors = np.zeros(1)  # np array with the index of corridors in each run

        self.get_lapdata(self.datapath, self.date_time, self.name, self.task,
                         selected_laps=self.selected_laps)  # collects all behavioral and imaging data and sort it into laps, storing each in a Lap_ImData object
        self.calc_active()
        if (self.n_laps == -1):
            print(
                'Error: missing laps are found in the ExpStateMachineLog file! No analysis was performed, check the logfiles!')
            return
        print('laps with im data: ', self.i_Laps_ImData)

        if not electrophysiology_analysis:
            frame_time_lap_maze_pos = np.array((self.frame_times, self.frame_laps, self.frame_maze, self.frame_pos))
            time_lap_maze_pos_FILE = self.suite2p_folder + 'frame_time_lap_maze_pos.npy'
            np.save(time_lap_maze_pos_FILE, frame_time_lap_maze_pos)

        ## in certain tasks, the same corridor may appear multiple times in different substages
        ## we need to keep this corridor in the list self.corridors for running self.get_lapdata()
        ## but we should remove the redundancy after the data is loaded
        self.all_corridors = np.unique(self.all_corridors)
        self.N_all_corridors = len(self.all_corridors)

        ## only analyse corridors with at least 3 laps
        # - the data still remains in the ImLaps list and will appear in the activity tensor!
        #   but the corridor will not
        #   we also do NOT include corridor 0 here
        if (self.N_all_corridors > 1):
            # print('i laps with imaging data:', self.i_Laps_ImData)
            # print('i of corridors: ', self.i_corridors)
            corridors, N_laps_corr = np.unique(self.i_corridors[self.i_Laps_ImData], return_counts=True)
            self.corridors = corridors[np.flatnonzero(N_laps_corr >= self.minimum_Nlaps)]
            self.N_corridors = len(self.corridors)
        else:
            self.corridors = np.setdiff1d(self.all_corridors, 0)
            self.N_corridors = len(self.corridors)

        self.N_ImLaps = len(self.i_Laps_ImData)
        # print('number of laps with imaging data:', self.N_ImLaps)
        self.raw_activity_tensor = np.zeros((self.N_pos_bins, self.N_cells,
                                             self.N_ImLaps))  # a tensor with space x neurons x trials containing the spikes
        self.raw_activity_tensor_time = np.zeros((self.N_pos_bins,
                                                  self.N_ImLaps))  # a tensor with space x trials containing the time spent at each location in each lap
        self.activity_tensor = np.zeros(
            (self.N_pos_bins, self.N_cells, self.N_ImLaps))  # same as the activity tensor spatially smoothed
        self.activity_tensor_time = np.zeros(
            (self.N_pos_bins, self.N_ImLaps))  # same as the activity tensor time spatially smoothed
        # print('laps with image data:')
        # print(self.i_Laps_ImData)
        self.combine_lapdata()  ## fills in the cell_activity tensor

        self.cell_activelaps = []  # a list, each element is a vector with the % of significantly spiking laps of the cells in a corridor
        self.cell_Fano_factor = []  # a list, each element is a vector with the reliability of the cells in a corridor

        self.cell_reliability = []  # a list, each element is a vector with the reliability of the cells in a corridor
        self.cell_skaggs = []  # a list, each element is a vector with the skaggs93 spatial info of the cells in a corridor
        self.cell_tuning_specificity = []  # a list, each element is a vector with the tuning specificity of the cells in a corridor

        self.cell_rates = []  # a list, each element is a 1 x n_cells matrix with the average rate of the cells in the whole corridor
        self.cell_corridor_selectivity = np.zeros([2,
                                                   self.N_cells])  # a matrix with the selectivity index of the cells. Second row indicates the corridor with the highers rate.

        self.ratemaps = []  # a list, each element is an array space x neurons x trials being the ratemaps of the cells in a given corridor
        self.cell_corridor_similarity = np.zeros(self.N_cells)  # a vector with the similarity index of the cells.

        self.calculate_properties()

        self.candidate_PCs = []  # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018
        self.accepted_PCs = []  # a list, each element is a vector of Trues and Falses of accepted place cells after bootstrapping
        self.Hainmuller_PCs()

        self.test_anticipatory()

    def test_anticipatory(self):
        self.anticipatory = []
        print('testing anticipatory licks...')
        print('N_corridors: ', self.N_corridors)
        print('corridors: ', self.corridors)

        for row in range(self.N_corridors):
            ids = np.flatnonzero(self.i_corridors == self.corridors[row])
            n_laps = len(ids)
            n_zones = np.shape(self.ImLaps[ids[0]].zones)[1]
            if (n_zones == 1):
                lick_rates = np.zeros([2,n_laps])
                k = 0
                for lap in np.nditer(ids):
                    if (self.ImLaps[lap].mode == 1):
                        lick_rates[:,k] = self.ImLaps[lap].preZoneRate
                    else:
                        lick_rates[:,k] = np.nan
                    k = k + 1
                self.anticipatory.append(AnticipatoryLicks(lick_rates[0,:], lick_rates[1,:], self.corridors[row]))

    def calculate_properties(self, nSD=4):
        self.cell_reliability = []
        self.cell_Fano_factor = []
        self.cell_skaggs=[]
        self.cell_activelaps=[]
        self.cell_activelaps_df=[]
        self.cell_tuning_specificity=[]

        self.cell_rates = [] # a list, each element is a 1 x n_cells matrix with the average rate of the cells in the whole corridor
        if (self.task == 'contingency_learning'):
            self.cell_pattern_rates = []
        self.ratemaps = [] # a list, each element is an array space x neurons x trials being the ratemaps of the cells in a given corridor

        self.cell_corridor_selectivity = np.zeros([2,self.N_cells]) # a matrix with the selectivity index of the cells. Second row indicates the corridor with the highers rate.
        self.cell_corridor_similarity = np.zeros(self.N_cells) # a matrix with the similarity index of the cells.

        if (self.N_corridors > 0):
            for i_corridor in np.arange(self.N_corridors): # we exclude corridor 0
                corridor = self.corridors[i_corridor]

                # select the laps in the corridor
                # only laps with imaging data are selected - this will index the activity_tensor
                i_laps = np.flatnonzero(self.i_corridors[self.i_Laps_ImData] == corridor)
                N_laps_corr = len(i_laps)

                time_matrix_1 = self.activity_tensor_time[:,i_laps]
                total_time = np.sum(time_matrix_1, axis=1) # bins x cells -> bins; time spent in each location

                act_tensor_1 = self.activity_tensor[:,:,i_laps] ## bin x cells x laps; all activity in all laps in corridor i
                total_spikes = np.sum(act_tensor_1, axis=2) ##  bin x cells; total activity of the selected cells in corridor i

                rate_matrix = np.zeros_like(total_spikes) ## event rate

                for i_cell in range(self.N_cells):
                # for i_cell in range(total_spikes.shape[1]):
                    rate_matrix[:,i_cell] = total_spikes[:,i_cell] / total_time
                self.ratemaps.append(rate_matrix)

                print('calculating rate, reliability and Fano factor...')
                ## average firing rate
                rates = np.sum(total_spikes, axis=0) / np.sum(total_time)
                self.cell_rates.append(rates)

                if (self.task == 'contingency_learning'):
                    zone_start = int(np.floor(self.ImLaps[0].zones[0]*self.N_pos_bins)) # 42-46 or 45-49
                    zone_end = int(np.floor(self.ImLaps[0].zones[1]*self.N_pos_bins))

                    rates_pattern1 = np.sum(total_spikes[0:14,:], axis=0) / np.sum(total_time[0:14])
                    rates_pattern2 = np.sum(total_spikes[14:28,:], axis=0) / np.sum(total_time[14:28])
                    rates_pattern3 = np.sum(total_spikes[28:42,:], axis=0) / np.sum(total_time[28:42])
                    rates_reward = np.sum(total_spikes[zone_start:zone_end,:], axis=0) / np.sum(total_time[zone_start:zone_end])
                    self.cell_pattern_rates.append(np.vstack([rates_pattern1, rates_pattern2, rates_pattern3, rates_reward]))

                ## reliability and Fano factor
                reliability = np.zeros(self.N_cells)
                Fano_factor = np.zeros(self.N_cells)
                for i_cell in range(self.N_cells):
                    laps_rates = nan_divide(act_tensor_1[:,i_cell,:], time_matrix_1, where=(time_matrix_1 > 0.025))
                    corrs_cell = vcorrcoef(np.transpose(laps_rates), rate_matrix[:,i_cell])
                    reliability[i_cell] = np.nanmean(corrs_cell)
                    Fano_factor[i_cell] = np.nanmean(nan_divide(np.nanvar(laps_rates, axis=1), rate_matrix[:,i_cell], rate_matrix[:,i_cell] > 0))
                self.cell_reliability.append(reliability)
                self.cell_Fano_factor.append(Fano_factor)


                print('calculating Skaggs spatial info...')
                ## Skaggs spatial info in bits per spike
                skaggs_vector=np.zeros(self.N_cells)
                P_x=total_time/np.sum(total_time)
                for i_cell in range(self.N_cells):
                    mean_firing = rates[i_cell]
                    lambda_x = rate_matrix[:,i_cell]
                    i_nonzero = np.nonzero(lambda_x > 0)
                    skaggs_vector[i_cell] = np.sum(lambda_x[i_nonzero]*np.log2(lambda_x[i_nonzero]/mean_firing)*P_x[i_nonzero]) / mean_firing
                self.cell_skaggs.append(skaggs_vector)

                ## active laps/ all laps spks
                #use raw spks instead activity tensor
                print('calculating proportion of active laps...')
                active_laps = np.zeros((self.N_cells, N_laps_corr))

                icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with image data
                i_laps_abs = self.i_Laps_ImData[np.nonzero(icorrids == corridor)[0]] # we need a different indexing here, for the ImLaps list and not fot  the activityTensor
                k = 0
                if (self.elfiz):
                    spike_threshold = 0.75
                else:
                    spike_threshold = 25

                for i_lap in i_laps_abs:#y=ROI
                    act_cells = np.nonzero(np.amax(self.ImLaps[i_lap].frames_spikes, 1) > spike_threshold)[0]
                    active_laps[act_cells, k] = 1
                    k = k + 1

                active_laps_ratio = np.sum(active_laps, 1) / N_laps_corr
                self.cell_activelaps.append(active_laps_ratio)

                ## dF/F active laps/all laps
                print('calculating proportion of active laps based on dF/F ...')
                active_laps_df = np.zeros((self.N_cells, N_laps_corr))
                k = 0
                for i_lap in i_laps_abs:
                    act_cells = np.nonzero(np.amax(self.ImLaps[i_lap].frames_dF_F, 1) > (self.cell_SDs*nSD))[0]
                    active_laps_df[act_cells, k] = 1
                    k = k + 1
                active_laps_ratio_df = np.sum(active_laps_df, 1) / N_laps_corr
                self.cell_activelaps_df.append(active_laps_ratio_df)

                ## linear tuning specificity
                print('calculating linear tuning specificity ...')
                tuning_spec = np.zeros(self.N_cells)
                xbins = (np.arange(self.N_pos_bins) + 0.5) * self.corridor_length_cm / self.N_pos_bins

                for i_cell in range(self.N_cells):
                    rr = np.copy(rate_matrix[:,i_cell])
                    rr[rr < np.mean(rr)] = 0
                    Px = rr / np.sum(rr)
                    mu = np.sum(Px * xbins)
                    sigma = np.sqrt(np.sum(Px * xbins**2) - mu**2)
                    tuning_spec[i_cell] = self.corridor_length_cm / sigma

                self.cell_tuning_specificity.append(tuning_spec)

        if (self.N_corridors > 1):
            self.calc_selectivity_similarity()

    def get_stage(self, action_log_file_path) -> int:
        # function that reads the action_log_file and finds the current stage
        with open(action_log_file_path, newline='') as action_log_file:
            log_file_reader = csv.reader(action_log_file, delimiter=',')
            next(log_file_reader, None)  # skip the headers
            for line in log_file_reader:
                if line[1] == 'Stage':
                    return int(round(float(line[2])))
            raise Exception(f"no \"Stage\" keyword in {action_log_file_path}")

    def LoadImaging_times(self, offset):
        imaging_logfile = minidom.parse(str(self.imaging_logfile_name))
        voltage_rec = imaging_logfile.getElementsByTagName('VoltageRecording')
        voltage_delay = float(voltage_rec[0].attributes['absoluteTime'].value)
        ## the offset is the time of the first voltage signal in Labview time
        ## the signal's 0 has a slight delay compared to the time 0 of the imaging recording
        ## we substract this delay from the offset to get the LabView time of the time 0 of the imaging recording
        corrected_offset = offset - voltage_delay
        print('corrected offset:', corrected_offset, 'voltage_delay:', voltage_delay)

        # find out whether it's a multiplane recording
        sequence = imaging_logfile.getElementsByTagName('Sequence')
        frames = imaging_logfile.getElementsByTagName('Frame')
        if sequence[0].attributes['type'].value == 'TSeries ZSeries Element':
            print('multi-plane')
            self.multiplane = True
            # for multiplane recordings we drop last frame as it is sometimes 'missing' for one of the planes
            self.F_all = self.F_all[:, 0:-1]
            self.spks_all = self.spks_all[:, 0:-1]
            # self.Fneu = self.Fneu[:, 0:-1]
            if len(frames) % 2 == 0:
                len_frames_used = int(len(frames) / 2 - 1)
            if len(frames) % 2 == 1:
                len_frames_used = int((len(frames) - 1) / 2)
            # for frame time we use the average of the two planes time
            self.frame_times = np.zeros(len_frames_used)
            self.im_reftime = float(frames[1].attributes['relativeTime'].value) - float(
                frames[1].attributes['absoluteTime'].value)
            for i in range(len_frames_used):  ## checkit: why i and i+1 and not 2i and 2i - 1?
                self.frame_times[i] = (float(frames[2 * i].attributes['relativeTime'].value) + float(
                    frames[2 * i + 1].attributes['relativeTime'].value)) / 2 + corrected_offset

        else:
            print('single-plane')

            self.frame_times = np.zeros(len(frames))  # this is already in labview time
            self.im_reftime = float(frames[1].attributes['relativeTime'].value) - float(
                frames[1].attributes['absoluteTime'].value)
            for i in range(len(frames)):
                self.frame_times[i] = float(frames[i].attributes['relativeTime'].value) + corrected_offset

        if (len(self.frame_times) != self.F_all.shape[1]):
            print('ERROR: imaging frame number does not match suite2p frame number! Something is wrong!')
            print('shape of the dF array:', self.F_all.shape)
            print('shape of spikes array:', self.spks_all.shape[1])
            print('suite2p frame number:', self.F_all.shape[1])
            print('imaging frame number:', len(self.frame_times))
            raise ValueError('ERROR: imaging frame number does not match suite2p frame number! Something is wrong!')

    def calc_dF_F(self):
        print('calculating dF/F and SNR...')

        self.cell_SDs = np.zeros(self.N_cells)  # a vector with the SD of the cells
        self.cell_SNR = np.zeros(self.N_cells)  # a vector with the signal to noise ratio of the cells (max F / SD)
        self.cell_baselines = np.zeros(self.N_cells)  # a vector with the baseline F of the cells

        ## to calculate the SD and SNR, we need baseline periods with no spikes for at least 1 sec
        self.frame_rate = int(np.ceil(1 / self.frame_period))
        sp_threshold = 20  #
        T_after_spike = 3  # s
        T_before_spike = 0.5  # s
        Tmin_no_spike = 1  # s
        L_after_spike = int(round(T_after_spike * self.frame_rate))
        L_before_spike = int(round(T_before_spike * self.frame_rate))
        Lmin_no_spike = int(round(Tmin_no_spike * self.frame_rate))

        N_frames = len(self.frame_times)
        filt = np.ones(self.frame_rate)

        # calculate baseline
        for i_cell in range(self.N_cells):

            # baseline: mode of the histogram
            trace = self.F[i_cell,]
            hist = np.histogram(trace, bins=100)
            max_index = np.where(hist[0] == max(hist[0]))[0][0]
            baseline = hist[1][max_index]
            # if (baseline == 0):
            #     baseline = hist[1][max_index+1]

            self.dF_F[i_cell,] = (self.F[i_cell,] - baseline) / baseline

            ### 1. find places where there are no spikes for a long interval
            ### 1.1. we add all spikes in a 1s window by convolving it with a 1s box car function
            # fig, ax = plt.subplots(figsize=(12,8))
            # i_plot = 0
            # cells = np.sort(np.random.randint(0, self.N_cells, 6))
            # cells[0] = 4
            # cells[5] = 1064

            allspikes_1s = np.hstack([np.repeat(sp_threshold, self.frame_rate - 1),
                                      np.convolve(self.raw_spks[i_cell, :], filt, mode='valid')])

            ### 1.2 no spikes if the sum remains smaller than sp_threshold
            sp_1s = np.copy(allspikes_1s)
            sp_1s[np.nonzero(allspikes_1s < sp_threshold)[0]] = 0

            ### 1.3. find silent sections
            rise_index = np.nonzero((sp_1s[0:-1] < 1) & (sp_1s[1:] >= 1))[0] + 1
            fall_index = np.nonzero((sp_1s[0:-1] > 1) & (sp_1s[1:] <= 1))[0] + 1
            if (len(rise_index) == 0):
                rise_index = np.array([int(len(sp_1s))])
            if (max(rise_index) < N_frames - 1000):
                rise_index = np.hstack([rise_index, int(len(sp_1s))])
            if (len(fall_index) == 0):
                fall_index = np.array([int(0)])

            # pairing rises with falls
            if (fall_index[0] > rise_index[0]):
                # print('deleting first rise')
                rise_index = np.delete(rise_index, 0)
            if (fall_index[-1] > rise_index[-1]):
                # print('deleting last fall')
                fall_index = np.delete(fall_index, -1)
            if (len(rise_index) != len(fall_index)):
                print('rise and fall could not be matched for cell ' + str(i_cell))

            long_index = np.nonzero((rise_index - fall_index) > L_after_spike + L_before_spike + Lmin_no_spike)[0]
            rise_ind = rise_index[long_index]
            fall_ind = fall_index[long_index]

            sds = np.zeros(len(rise_ind))
            bases = np.zeros(len(rise_ind))
            for k in range(len(rise_ind)):
                i_start = fall_ind[k] + L_after_spike
                i_end = rise_ind[k] - L_before_spike
                sds[k] = np.sqrt(np.var(self.dF_F[i_cell, i_start:i_end]))
                bases[k] = np.average(self.dF_F[i_cell, i_start:i_end])
            self.cell_baselines[i_cell] = np.mean(bases)
            self.cell_SDs[i_cell] = np.mean(sds)
            self.cell_SNR[i_cell] = max(self.dF_F[i_cell, :]) / np.mean(sds)
            self.spks[i_cell, :] = self.spks[i_cell, :]  # / baseline / self.cell_SDs[i_cell]

        print('SNR done')

        print('dF/F calculated for cell ROI-s')

    def detect_events(self, sd_times=3): \
            # detecting significant events in the fluorescence signal
        # an event is significant, if the Gaussian filtered (SD = 3 x Interframe interval ) dF/F
        #
        # this function is redundant with the calc_active function in some parts. We need this in order to be able to pass events to individual laps
        sdfilt = 3
        N = 10
        sampling_time = 1
        xfilt = np.arange(-N * sdfilt, N * sdfilt + sampling_time, sampling_time)
        filt = np.exp(-(xfilt ** 2) / (2 * (sdfilt ** 2)))
        filt = filt / sum(filt)
        self.events = np.zeros(self.F.shape)

        for i in range(self.N_cells):
            temp = np.hstack(
                [np.repeat(self.dF_F[i, 0], N * sdfilt), self.dF_F[i, :], np.repeat(self.dF_F[i, -1], N * sdfilt)])
            dF_F_s = np.convolve(temp, filt, mode='valid')
            threshold = self.cell_baselines[i] + self.cell_SDs[i] * sd_times
            rises = np.nonzero((dF_F_s[0:-1] < threshold) & (dF_F_s[1:] >= threshold))[0]

            self.events[i, rises] = 1  # here we do not take refractoriness into account

    def get_lapdata(self, datapath, date_time, name, task, selected_laps=None):

        time_array = []
        lap_array = []
        maze_array = []
        position_array = []
        mode_array = []
        lick_array = []
        action = []
        substage = []

        # this could be made much faster by using pandas
        data_log_file_string = datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_' + name + '_' + task + '_ExpStateMashineLog.txt'
        data_log_file = open(data_log_file_string, newline='')
        log_file_reader = csv.reader(data_log_file, delimiter=',')
        next(log_file_reader, None)  # skip the headers
        for line in log_file_reader:
            time_array.append(float(line[0]))
            lap_array.append(int(line[1]))
            maze_array.append(int(line[2]))
            position_array.append(int(line[3]))
            mode_array.append(line[6] == 'Go')
            lick_array.append(line[9] == 'TRUE')
            action.append(str(line[14]))
            substage.append(str(line[17]))

        laptime = np.array(time_array)
        time_breaks = np.where(np.diff(laptime) > 1)[0]
        if (len(time_breaks) > 0):
            print('ExpStateMachineLog time interval > 1s: ', len(time_breaks), ' times')
            # print(laptime[time_breaks])

        lap = np.array(lap_array)
        logged_laps = np.unique(lap)
        all_laps = np.arange(max(lap)) + 1
        missing_laps = np.setdiff1d(all_laps, logged_laps)

        if (len(missing_laps) > 0):
            print('Some laps are not logged. Number of missing laps: ', len(missing_laps))
            print(missing_laps)
            self.n_laps = -1
            return

        sstage = np.array(substage)
        current_sstage = sstage[0]

        pos = np.array(position_array)
        lick = np.array(lick_array)
        maze = np.array(maze_array)
        mode = np.array(mode_array)

        #################################################
        ## add position, and lap info into the imaging frames
        #################################################
        F = interp1d(laptime, pos)
        self.frame_pos = np.round(F(self.frame_times), 2)
        F = interp1d(laptime, lap)
        self.frame_laps = F(self.frame_times)
        F = interp1d(laptime, maze)
        self.frame_maze = F(self.frame_times)

        print('length of frame_times:', len(self.frame_times))
        print('length of frame_laps:', len(self.frame_laps))
        print('shape of dF_F:', np.shape(self.dF_F))

        self.all_corridor_start_time = []
        self.all_corridor_start_IDs = []
        ## frame_laps is NOT integer for frames between laps
        ## however, it MAY not be integer even even for frames within a lap...
        #################################################
        ## end of - add position, and lap info into the imaging frames
        #################################################
        i_ImData = []  # index of laps with imaging data
        i_corrids = []  # ID of corridor for the current lap

        self.n_laps = 0  # counting only the laps loaded for analysis
        lap_count = 0  # counting all laps except grey zone
        N_0lap = 0  # counting the non-valid laps

        grey_zone_active = False
        if (np.unique(maze)[0] == 0):
            grey_zone_active = True
            # print('grey zone is active')
            # grey_zone_duration = []
            # correct_error = []
        if self.elfiz:
            imaging_min_position = 200  # every valid lap has to have a position corresponding to a frame that is lower than this at the begining (end also checked)
        else:
            imaging_min_position = self.corridor_length_roxel / (
                        7 / 8 * self.frame_rate)  # every valid lap has to have a position corresponding to a frame that is lower than this at the begining (end also checked)

        for i_lap in np.unique(lap):
            y = np.flatnonzero(lap == i_lap)  # index for the current lap

            mode_lap = np.prod(mode[y])  # 1 if all elements are recorded in 'Go' mode

            maze_lap = np.unique(maze[y])
            if (len(maze_lap) == 1):
                corridor = self.all_corridors[
                    maze_lap[0]]  # the maze_lap is the index of the available corridors in the given stage
            else:
                corridor = -1
            # print('corridor in lap ', self.n_laps, ':', corridor)

            sstage_lap = np.unique(sstage[y])

            if (len(sstage_lap) > 1):
                print('More than one substage in a lap before lap ', self.n_laps, 'in corridor', corridor)
                corridor = -2

            if (corridor > 0):
                if (y.size < self.N_pos_bins):
                    print('Very short lap found, we have total ', len(y),
                          'datapoints recorded by the ExpStateMachine in a lap before lap', self.n_laps, 'in corridor',
                          corridor)
                    corridor = -3

            if (corridor > 0):
                pos_lap = pos[y]
                n_posbins = len(np.unique(pos_lap))
                if (n_posbins < (self.corridor_length_roxel * 0.9)):
                    print('Short lap found, we have total ', n_posbins,
                          'position bins recorded by the ExpStateMachine in a lap before lap', self.n_laps,
                          'in corridor', corridor)
                    corridor = -4

                # if (min(pos_lap) > 10):
                #     print('Late-start lap found, first position:', np.min(pos_lap), 'in lap', self.n_laps, 'in corridor', corridor)

                # if (max(pos_lap) < (self.corridor_length_roxel - 10)):
                #     print('Early-end lap found, last position:', np.max(pos_lap), 'in lap', self.n_laps, 'in corridor', corridor)

            # print('processing corridor', corridor, 'in lap', i_lap)

            t_lap = laptime[y]
            next_grey_lap_duration = None
            self.all_corridor_start_time.append(min(t_lap))
            self.all_corridor_start_IDs.append(int(corridor))

            if (corridor > 0):
                # if we select laps, then we check lap ID:
                if (selected_laps is None):
                    add_lap = True
                else:
                    if (np.isin(lap_count, selected_laps)):
                        add_lap = True
                    else:
                        add_lap = False

                if (add_lap):
                    i_corrids.append(corridor)  # list with the index of corridors in each run
                    pos_lap = pos[y]

                    lick_lap = lick[y]  ## vector of Trues and Falses
                    t_licks = t_lap[lick_lap]  # time of licks

                    if (sstage_lap != current_sstage):
                        print('############################################################')
                        print('substage change detected!')
                        print('first lap in substage ', sstage_lap, 'is lap', self.n_laps, ', which started at t',
                              t_lap[0])
                        print('the time of the change in imaging time is: ', t_lap[0] - self.imstart_time)
                        print('############################################################')
                        current_sstage = sstage_lap
                        self.substage_change_laps.append(self.n_laps)
                        self.substage_change_time.append(t_lap[0] - self.imstart_time)

                    istart = np.min(y)  # action is not a np.array, array indexing does not work
                    iend = np.max(y) + 1
                    action_lap = action[istart:iend]

                    reward_indices = [j for j, x in enumerate(action_lap) if x == "TrialReward"]
                    t_reward = t_lap[reward_indices]

                    ## detecting invalid laps - terminated before the animal could receive reward
                    valid_lap = False
                    if (len(t_reward) > 0):  # lap is valid if the animal got reward
                        valid_lap = True
                        # print('rewarded lap', self.n_laps)
                    if (max(pos_lap) > (
                            self.corridor_length_roxel * self.last_zone_end)):  # # lap is valid if the animal left the last reward zone
                        valid_lap = True
                        # print('valid lap', self.n_laps)
                    if (valid_lap == False):
                        mode_lap = 0
                        # print('invalid lap', self.n_laps)

                    if (
                    grey_zone_active):  # we calculate the duration of the next grey zone - we can use this to double check rewarded laps
                        y_g = np.flatnonzero(lap == i_lap + 1)  # index for the current lap
                        if (len(y_g) > 1):
                            maze_lap_g = np.unique(maze[y_g])
                            if (len(maze_lap_g) == 1):
                                corridor_g = self.all_corridors[
                                    int(maze_lap_g)]  # the maze_lap is the index of the available corridors in the given stage
                            if (corridor_g != 0):
                                print('no next grey zone found')
                            else:
                                t_g_lap = laptime[y_g]
                                next_grey_lap_duration = np.max(t_g_lap) - np.min(t_g_lap)

                    actions = []
                    for j in range(len(action_lap)):
                        if not ((action_lap[j]) in ['No', 'TrialReward']):
                            actions.append([t_lap[j], action_lap[j]])

                    ## include only valid laps
                    add_ImLap = True
                    if (mode_lap == 0):
                        add_ImLap = False
                        print('lap mode = 0')

                    ### imaging data
                    iframes = np.flatnonzero(self.frame_laps == i_lap)
                    if (len(iframes) > 1):  # there is imaging data belonging to this lap...
                        lap_frames_dF_F = self.dF_F[:, iframes]
                        lap_frames_spikes = self.spks[:, iframes]
                        lap_frames_time = self.frame_times[iframes]
                        lap_frames_pos = self.frame_pos[iframes]
                        lap_frames_events = self.events[:, iframes]
                        # print(self.n_laps, np.min(lap_frames_pos), np.max(lap_frames_pos))
                        if (np.min(lap_frames_pos) > imaging_min_position):
                            add_ImLap = False
                            print('Late-start lap found, first position:', np.min(lap_frames_pos), 'in lap',
                                  self.n_laps, 'in corridor', corridor)
                        if (np.max(lap_frames_pos) < (self.corridor_length_roxel - imaging_min_position)):
                            add_ImLap = False
                            print('Early end lap found, last position:', np.max(lap_frames_pos), 'in lap', self.n_laps,
                                  'in corridor', corridor)
                    else:
                        add_ImLap = False

                    if (add_ImLap):  # there is imaging data belonging to this lap...
                        i_ImData.append(self.n_laps)
                        # print('frames:', min(iframes), max(iframes))
                        # print('max of iframes:', max(iframes))
                    else:
                        lap_frames_dF_F = np.nan
                        lap_frames_spikes = np.nan
                        lap_frames_time = np.nan
                        lap_frames_pos = np.nan
                        lap_frames_events = np.nan
                        # print('In lap ', self.n_laps, ' we have ', len(t_lap), 'datapoints and ', len(iframes), 'frames')

                    self.ImLaps.append(
                        Lap_ImData(self.name, self.n_laps, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap,
                                   actions, lap_frames_dF_F, lap_frames_spikes, lap_frames_pos, lap_frames_time,
                                   self.corridor_list, lap_frames_events, self.frame_period,
                                   speed_threshold=self.speed_threshold, elfiz=self.elfiz, multiplane=self.multiplane,
                                   next_grey_lap_duration=next_grey_lap_duration))
                    self.n_laps = self.n_laps + 1
                    lap_count = lap_count + 1
                else:
                    # print('lap ', lap_count, ' skipped.')
                    lap_count = lap_count + 1
                    # print(self.n_laps)
            else:
                # if (corridor == 0):
                #     lap_duration = np.max(t_lap) - np.min(t_lap)
                #     previous_correct = 'first lap'
                #     if ((lap_count > 0) & (corridor == 0)):
                #         previous_correct = self.ImLaps[self.n_laps-1].correct
                #         print('duration of grey zone after lap', self.n_laps-1, ': ', lap_duration, previous_correct)
                #         correct_error.append(previous_correct)
                #         grey_zone_duration.append(lap_duration)
                N_0lap = N_0lap + 1  # grey zone (corridor == 0) or invalid lap (corridor = -1) - we do not do anything with this...

        # correct_error = np.array(correct_error)
        # i_correct = np.flatnonzero(correct_error == 1)
        # i_error = np.flatnonzero(correct_error == 0)
        # grey_zone_duration = np.array(grey_zone_duration)
        # print('minimum grey zone after error:', np.min(grey_zone_duration[i_error]), ', max grey zone after correct:', np.max(grey_zone_duration[i_correct]))

        self.i_Laps_ImData = np.array(i_ImData)  # index of laps with imaging data
        self.i_corridors = np.array(i_corrids)  # ID of corridor for the current lap


def locate_imaging(trigger_log_file_path, trigger_voltage_path):

    trigger_log_starts = []  ## s
    trigger_log_lengths = []
    with open(trigger_log_file_path, newline='') as trigger_log_file:
        log_file_reader = csv.reader(trigger_log_file, delimiter=',')
        next(log_file_reader, None)  # skip the headers
        for line in log_file_reader:
            trigger_log_starts.append(float(line[0]))  # seconds
            trigger_log_lengths.append(float(line[1]) / 1000)  # convert to seconds from ms

        trigger_starts = np.array(trigger_log_starts)
        trigger_lengths = np.array(trigger_log_lengths)

    TRIGGER_VOLTAGE_VALUE = []
    TRIGGER_VOLTAGE_TIMES = []

    with open(trigger_voltage_path, newline='') as trigger_signal_file:
    ## ms
        trigger_reader = csv.reader(trigger_signal_file, delimiter=',')
        next(trigger_reader, None)
        for line in trigger_reader:
            TRIGGER_VOLTAGE_VALUE.append(float(line[1]))
            TRIGGER_VOLTAGE_TIMES.append(float(line[0]) / 1000)  # converting it to seconds
        TRIGGER_VOLTAGE = np.array(TRIGGER_VOLTAGE_VALUE)
        TRIGGER_TIMES = np.array(TRIGGER_VOLTAGE_TIMES)

    ## find trigger start and end times
    rise_index = np.nonzero((TRIGGER_VOLTAGE[0:-1] < 1) & (TRIGGER_VOLTAGE[1:] >= 1))[
                     0] + 1  # +1 needed otherwise we are pointing to the index just before the trigger
    RISE_T = TRIGGER_TIMES[rise_index]

    fall_index = np.nonzero((TRIGGER_VOLTAGE[0:-1] > 1) & (TRIGGER_VOLTAGE[1:] <= 1))[0] + 1
    FALL_T = TRIGGER_TIMES[fall_index]

    # pairing rises with falls
    if (RISE_T[0] > FALL_T[0]):
        FALL_T = np.delete(FALL_T, 0)


    if (RISE_T[-1] > FALL_T[-1]):
        RISE_T = np.delete(RISE_T, -1)


    if np.size(RISE_T) != np.size(FALL_T):
        print('rises:', np.size(RISE_T), 'falls:', np.size(FALL_T))
        raise Exception('trigger ascending and desending edges do not match! unable to locate imaging part')

    # 1) filling up TRIGGER_DATA array:
    # TRIGGER_DATA: 0. start time, 1. end time, 2. duration, 3.ITT, 4. index
    TRIGGER_DATA = np.zeros((np.size(RISE_T), 5))
    TRIGGER_DATA[:, 0] = RISE_T
    TRIGGER_DATA[:, 1] = FALL_T
    TRIGGER_DATA[:, 2] = FALL_T - RISE_T  # duration
    TEMP_FALL = np.concatenate([[0], FALL_T])
    TEMP_FALL = np.delete(TEMP_FALL, -1)
    TRIGGER_DATA[:, 3] = RISE_T - TEMP_FALL  # previous down duration - Inter Trigger Time
    TRIGGER_DATA[:, 4] = np.arange(0, np.size(RISE_T))

    # 2) keeping only triggers with ITT > 10 ms
    valid_indexes = np.nonzero(TRIGGER_DATA[:, 3] > 0.010)[0]
    TRIGGER_DATA_sub = TRIGGER_DATA[valid_indexes, :]

    # 3) find the valid shortest trigger
    lengths = np.copy(TRIGGER_DATA_sub[:, 2])
    index = 0
    if lengths.size < 2:
        raise Exception('Less than 2 valid triggers - Unable to locate imaging!')
    if lengths.size < 6:
        used_index = int(TRIGGER_DATA_sub[0][4])
        n_extra_indexes = min(5, TRIGGER_DATA.shape[0] - used_index)
    else:
        while True:
            minindex = np.argmin(lengths)
            used_index = int(TRIGGER_DATA_sub[minindex][4])
            n_extra_indexes = min(5, TRIGGER_DATA.shape[0] - used_index)
            if n_extra_indexes < 5:
                lengths[minindex] = np.max(lengths) + 1
                index += 1
                if index == lengths.size:
                    raise Exception('Unable to locate imaging! Not enough checkable valid triggers!')
            else:
                break

    # 4)find the candidate trigger times
    candidate_log_indexes = []
    for i in range(len(trigger_lengths)):
        if (abs(trigger_lengths[i] - TRIGGER_DATA[used_index][2]) < 0.007):
            candidate_log_indexes.append(i)

    if TRIGGER_DATA[used_index, 2] > 0.800:
        raise Exception('Warning! No short enough trigger in this recording! Unable to locate imaging')

    match_found = False

    for i in range(len(candidate_log_indexes)):
        log_reference_index = candidate_log_indexes[i]
        difs = []
        if len(trigger_starts) > log_reference_index + n_extra_indexes:
            for j in range(n_extra_indexes):
                dif_log = trigger_starts[log_reference_index + j] - trigger_starts[log_reference_index]
                dif_mes = TRIGGER_DATA[used_index + j, 0] - TRIGGER_DATA[used_index, 0]
                delta = abs(dif_log - dif_mes)
                difs.append(delta)

            if max(difs) < 0.009:
                return trigger_starts[log_reference_index] - TRIGGER_DATA[used_index, 0]
        else:
            print('   slight warning - testing some late candidates failed')

    raise Exception('no precise trigger mach found: need to refine code or check device')



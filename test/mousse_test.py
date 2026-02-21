import os
import pathlib

from mousse.image_session_data import ImagingSessionData


datapath = pathlib.Path(__file__).parents[1] / 'data' #current working directory - look for data and strings here!
date_time = '2021-10-31_10-31-32' # date and time of the imaging session
name = 'KS030' # mouse name
task = 'NearFarLong' # task name

## locate the suite2p folder
suite2p_folder = datapath / (name + '_imaging') / 'KS030_103121'

## the name and location of the imaging log file
imaging_logfile_name = suite2p_folder / 'KS030_TSeries-10312021-1017-001.xml'

## the name and location of the trigger voltage file
trigger_voltage_path = suite2p_folder / 'KS030_TSeries-10312021-1017-001_Cycle00001_VoltageRecording_001.csv'
action_log_file_path = datapath / 'KS030_NearFarLong' / date_time / '2021-10-31_10-31-32_KS030_NearFarLong_UserActionLog.txt'
trigger_log_file_path = datapath / 'KS030_NearFarLong' / date_time / '2021-10-31_10-31-32_KS030_NearFarLong_TriggerLog.txt'
F_all_path = suite2p_folder / 'F.npy'
spikes_all_path = suite2p_folder / 'spks.npy'
is_cell_path = suite2p_folder / 'iscell.npy'

D1 = ImagingSessionData(
    datapath,
    date_time,
    name,
    task,
    suite2p_folder,
    imaging_logfile_name,
    trigger_voltage_path=trigger_voltage_path,
    action_log_file_path=action_log_file_path,
    trigger_log_file_path=trigger_log_file_path,
    F_all_path=F_all_path,
    spikes_all_path=spikes_all_path,
    is_cell_path=is_cell_path,
)

import numpy as np
import pandas as pd

# == Import data from csv ==

# PROSIT
chunks_prosit = pd.read_csv("export_prosit.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'env_illuminance', 'env_illuminance_d', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate']]
chunks_prosit['subject_id'] = chunks_prosit['session_description'].str.split('-').str.get(0)
sessions_prosit = chunks_prosit.drop_duplicates(subset='session_id')
# VV
chunks_vv = pd.read_csv("export_vv_medium.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'env_illuminance', 'env_illuminance_d', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate']]
chunks_vv['subject_id'] = chunks_vv['session_description'].str.split('/').str.get(1)
sessions_vv = chunks_vv.drop_duplicates(subset='session_id')
# VV-Ghana
chunks_vv_ghana = pd.read_csv("export_vv_ghana_small.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'env_illuminance', 'env_illuminance_d', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate']]
chunks_vv_ghana['subject_id'] = chunks_vv_ghana['session_description'].str.split('/').str.get(1)
sessions_vv_ghana = chunks_vv_ghana.drop_duplicates(subset='session_id')

# == Dataset summaries ==

# PROSIT
splits = ['train', 'valid', 'test']
prosit_summary = pd.DataFrame({
  'split': [s.capitalize() for s in splits],
  'participants': [chunks_prosit[chunks_prosit['session_split'] == s]['subject_id'].nunique() for s in splits],
  'sessions': [chunks_prosit[chunks_prosit['session_split'] == s]['session_description'].nunique() for s in splits],
  'chunks': [chunks_prosit[chunks_prosit['session_split'] == s].shape[0] for s in splits],
  'time': [round(chunks_prosit[chunks_prosit['session_split'] == s]['duration'].sum()/(60.*60.), 1) for s in splits],
})
prosit_summary.to_csv("data/prosit_summary.csv", index=False)
# VV-Medium
vv_medium_summary = pd.DataFrame({
  'split': ['Test'],
  'participants': [int(chunks_vv['subject_id'].nunique())],
  'chunks': [chunks_vv.shape[0]],
  'time': [round(chunks_vv['duration'].sum()/(60.*60.), 1)],
})
vv_medium_summary.to_csv("data/vv_medium_summary.csv", index=False)
# VV-Ghana-Small
vv_ghana_small_summary = pd.DataFrame({
  'split': [s.capitalize() for s in splits],
  'participants': [chunks_vv_ghana[chunks_vv_ghana['session_split'] == s]['subject_id'].nunique() for s in splits],
  'chunks': [chunks_vv_ghana[chunks_vv_ghana['session_split'] == s].shape[0] for s in splits],
  'time': [round(chunks_vv_ghana[chunks_vv_ghana['session_split'] == s]['duration'].sum()/(60.*60.), 1) for s in splits],
})
vv_ghana_small_summary.to_csv("data/vv_ghana_small_summary.csv", index=False)
# Training data)
sources = ['PROSIT', 'VV-Ghana-Small']
training_summary = pd.DataFrame({
  'source': sources,
  'participants': [prosit_summary[prosit_summary['split'] == 'Train']['participants'].values[0] if s == 'PROSIT' else vv_ghana_small_summary[vv_ghana_small_summary['split'] == 'Train']['participants'].values[0] for s in sources],
  'chunks': [prosit_summary[prosit_summary['split'] == 'Train']['chunks'].values[0] if s == 'PROSIT' else vv_ghana_small_summary[vv_ghana_small_summary['split'] == 'Train']['chunks'].values[0] for s in sources],
  'time': [prosit_summary[prosit_summary['split'] == 'Train']['time'].values[0] if s == 'PROSIT' else vv_ghana_small_summary[vv_ghana_small_summary['split'] == 'Train']['time'].values[0] for s in sources],
})
training_summary.to_csv("data/training_summary.csv", index=False)

# == Demographics ==

# Training Dataset Age
frequencies, bin_edges = np.histogram(np.concatenate([sessions_prosit[sessions_prosit['session_split'] == 'train']['subject_age'],
                                                      sessions_vv_ghana[sessions_vv_ghana['session_split'] == 'train']['subject_age']], axis=-1),
                                      bins=8)
np.savetxt('data/age_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# Training Dataset Gender
counts_prosit = sessions_prosit[sessions_prosit['session_split'] == 'train']['subject_gender'].value_counts()
counts_vv_ghana = sessions_vv_ghana[sessions_vv_ghana['session_split'] == 'train']['subject_gender'].value_counts()
counts = counts_prosit.add(counts_vv_ghana, fill_value=0).astype(int)
counts_df = pd.DataFrame({'Label': counts.index.str.capitalize(), 'Value': counts.values})
counts_df.to_csv("data/gender.csv", index=False)

# Training Dataset Skin Type
counts_prosit = sessions_prosit[sessions_prosit['session_split'] == 'train']['subject_skin_type'].value_counts()
counts_vv_ghana = sessions_vv_ghana[sessions_vv_ghana['session_split'] == 'train']['subject_skin_type'].value_counts()
counts = counts_prosit.add(counts_vv_ghana, fill_value=0).astype(int)
counts_df = pd.DataFrame({'Label': counts.index.map(lambda val: "Type {}".format(val)), 'Value': counts.values})
counts_df = counts_df.sort_values(by='Label')
counts_df.to_csv("data/skin_type.csv", index=False)

# == Summary vitals ==

# Training Dataset HR
frequencies, bin_edges = np.histogram(np.concatenate([chunks_prosit[chunks_prosit['session_split'] == 'train']['frame_avg_hr_pox'],
                                                      chunks_vv_ghana[chunks_vv_ghana['session_split'] == 'train']['frame_avg_hr_pox']], axis=-1),
                                      bins=10)
np.savetxt('data/hr_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# Training Dataset RR
frequencies, bin_edges = np.histogram(np.concatenate([chunks_prosit[(chunks_prosit['session_split'] == 'train') & (chunks_prosit['frame_avg_rr'] >= 0) & (chunks_prosit['frame_avg_rr'] <= 40)]['frame_avg_rr'],
                                                      chunks_vv_ghana[chunks_vv_ghana['session_split'] == 'train']['frame_avg_rr']], axis=-1),
                                      bins=10)
np.savetxt('data/rr_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# Training Dataset BP
frequencies_sys, bin_edges_sys = np.histogram(np.concatenate([chunks_prosit[chunks_prosit['session_split'] == 'train']['frame_avg_bp_sys'],
                                                              chunks_vv_ghana[chunks_vv_ghana['session_split'] == 'train']['frame_avg_bp_sys']], axis=-1),
                                              bins=10)
frequencies_dia, bin_edges_dia = np.histogram(np.concatenate([chunks_prosit[chunks_prosit['session_split'] == 'train']['frame_avg_bp_dia'],
                                                              chunks_vv_ghana[chunks_vv_ghana['session_split'] == 'train']['frame_avg_bp_dia']], axis=-1),
                                              bins=10)
np.savetxt('data/bp_sys_histogram.csv', np.column_stack(((bin_edges_sys[:-1]+bin_edges_sys[1:])/2, frequencies_sys)), delimiter=',', header='BinEdges,Frequency', comments='')
np.savetxt('data/bp_dia_histogram.csv', np.column_stack(((bin_edges_dia[:-1]+bin_edges_dia[1:])/2, frequencies_dia)), delimiter=',', header='BinEdges,Frequency', comments='')

# Training Dataset SpO2
frequencies, bin_edges = np.histogram(np.concatenate([chunks_prosit[(chunks_prosit['session_split'] == 'train') & (chunks_prosit['frame_avg_spo2'] >= 50)]['frame_avg_spo2'],
                                                      chunks_vv_ghana[(chunks_vv_ghana['session_split'] == 'train') & (chunks_vv_ghana['frame_avg_spo2'] >= 50)]['frame_avg_spo2']], axis=-1),
                                      bins=10)
np.savetxt('data/spo2_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# == Results ==

# Import for PROSIT
# TODO: This is actually valid for now. Evaluate for test.
def merge_force_suffix(left, right, **kwargs):
  left_on_col = kwargs['left_on']
  right_on_col = kwargs['right_on']
  suffix_tupple = kwargs['suffixes']
  def suffix_col(col, suffix):
    if col != left_on_col and col != right_on_col:
      return str(col) + suffix
    else:
      return col
  left_suffixed = left.rename(columns=lambda x: suffix_col(x, suffix_tupple[0]))
  right_suffixed = right.rename(columns=lambda x: suffix_col(x, suffix_tupple[1]))
  del kwargs['suffixes']
  return pd.merge(left_suffixed, right_suffixed, **kwargs)
raw_plethnet_prosit = pd.read_csv("evaluate_plethnet_v2_72_prosit_30.csv")
raw_plethnet_prosit = raw_plethnet_prosit[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
results_prosit_test = merge_force_suffix(chunks_prosit, raw_plethnet_prosit, left_on='id', right_on='chunk_id', how='inner', suffixes=['', '_vl'])
raw_pos_prosit = pd.read_csv("evaluate_plethnet_v2_72_prosit_15.csv")
raw_pos_prosit = raw_pos_prosit[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
results_prosit_test = merge_force_suffix(results_prosit_test, raw_pos_prosit, left_on='id', right_on='chunk_id', how='inner', suffixes=['', '_pos'])

# Import for VV
raw_plethnet_vv = pd.read_csv("evaluate_plethnet_v2_72_vv_30.csv")
raw_plethnet_vv = raw_plethnet_vv[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
results_vv = merge_force_suffix(pd.concat([chunks_vv, chunks_vv_ghana], ignore_index=True), raw_plethnet_vv, left_on='id', right_on='chunk_id', how='inner', suffixes=['', '_vl'])
raw_pos_vv = pd.read_csv("evaluate_plethnet_v2_72_vv_15.csv")
raw_pos_vv = raw_pos_vv[['chunk_id', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
results_vv = merge_force_suffix(results_vv, raw_pos_vv, left_on='id', right_on='chunk_id', how='inner', suffixes=['', '_pos'])

#methods = ['G', 'CHROM', 'POS', 'DeepPhys', 'MTTS-CAN', 'VitalLens']
#methods_print = ['g', 'chrom', 'pos', 'deepphys', 'mtts', 'vl']
methods = ['pos', 'vl']
methods_print = ['POS', 'VitalLens']

## VV-Medium
results_vv_medium = results_vv[~results_vv['session_description'].str.contains('ghana')]
results_vv_medium = pd.DataFrame({
  'method': methods_print,
  'hr_mae': [results_vv_medium['hr_ae_{}'.format(m)].mean() for m in methods],
  'pulse_snr': [results_vv_medium['pulse_snr_{}'.format(m)].mean() for m in methods],
  'pulse_cor': [results_vv_medium['pulse_cor_{}'.format(m)].mean() for m in methods],
  'rr_mae': [results_vv_medium['rr_ae_{}'.format(m)].mean() for m in methods],
  'resp_snr': [results_vv_medium['resp_snr_{}'.format(m)].mean() for m in methods],
  'resp_cor': [results_vv_medium['resp_cor_{}'.format(m)].mean() for m in methods],
})
results_vv_medium.to_csv("data/results_vv_medium.csv", index=False)

## VV-Ghana Test Set
results_vv_ghana_test = results_vv[(results_vv['session_description'].str.contains('ghana')) & (results_vv['session_split'] == 'test')]
results_vv_ghana_test = pd.DataFrame({
  'method': methods_print,
  'hr_mae': [results_vv_ghana_test['hr_ae_{}'.format(m)].mean() for m in methods],
  'pulse_snr': [results_vv_ghana_test['pulse_snr_{}'.format(m)].mean() for m in methods],
  'pulse_cor': [results_vv_ghana_test['pulse_cor_{}'.format(m)].mean() for m in methods],
  'rr_mae': [results_vv_ghana_test['rr_ae_{}'.format(m)].mean() for m in methods],
  'resp_snr': [results_vv_ghana_test['resp_snr_{}'.format(m)].mean() for m in methods],
  'resp_cor': [results_vv_ghana_test['resp_cor_{}'.format(m)].mean() for m in methods],
})
results_vv_ghana_test.to_csv("data/results_vv_ghana_test.csv", index=False)

## PROSIT Test Set
results_prosit_test = pd.DataFrame({
  'method': methods_print,
  'hr_mae': [results_prosit_test['hr_ae_{}'.format(m)].mean() for m in methods],
  'pulse_snr': [results_prosit_test['pulse_snr_{}'.format(m)].mean() for m in methods],
  'pulse_cor': [results_prosit_test['pulse_cor_{}'.format(m)].mean() for m in methods],
  'rr_mae': [results_prosit_test['rr_ae_{}'.format(m)].mean() for m in methods],
  'resp_snr': [results_prosit_test['resp_snr_{}'.format(m)].mean() for m in methods],
  'resp_cor': [results_prosit_test['resp_cor_{}'.format(m)].mean() for m in methods],
})
results_prosit_test.to_csv("data/results_prosit_test.csv", index=False)

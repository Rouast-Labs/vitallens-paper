import numpy as np
import pandas as pd

# == Import data from csv ==

# PROSIT
chunks_prosit = pd.read_csv("export_prosit.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type']]
chunks_prosit['subject_id'] = chunks_prosit['session_description'].str.split('-').str.get(0)
sessions_prosit = chunks_prosit.drop_duplicates(subset='session_id')
# VV
chunks_vv = pd.read_csv("export_vv_medium.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type']]
chunks_vv['subject_id'] = chunks_vv['session_description'].str.split('/').str.get(1)
sessions_vv = chunks_vv.drop_duplicates(subset='session_id')
# VV-Ghana
chunks_vv_ghana = pd.read_csv("export_vv_ghana_small.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type']]
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

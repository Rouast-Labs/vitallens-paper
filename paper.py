import numpy as np
import pandas as pd

export_chunks = pd.read_csv("export_chunks.csv")
chunks = export_chunks[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type']]
sessions = chunks.drop_duplicates(subset='session_id')

# Age
frequencies, bin_edges = np.histogram(sessions['subject_age'], bins=8)
np.savetxt('data/age_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# Gender
counts = sessions['subject_gender'].value_counts()
counts_df = pd.DataFrame({'Label': counts.index.str.capitalize(), 'Value': counts.values})
counts_df.to_csv("data/gender.csv", index=False)

# Skin Type
counts = sessions['subject_skin_type'].value_counts()
counts_df = pd.DataFrame({'Label': counts.index.map(lambda val: "Type {}".format(val)), 'Value': counts.values})
counts_df = counts_df.sort_values(by='Label')
counts_df.to_csv("data/skin_type.csv", index=False)

# HR
frequencies, bin_edges = np.histogram(chunks['frame_avg_hr_pox'], bins=10)
np.savetxt('data/hr_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# RR
frequencies, bin_edges = np.histogram(chunks[(chunks['frame_avg_rr'] >= 0) & (chunks['frame_avg_rr'] <= 40)]['frame_avg_rr'], bins=10)
np.savetxt('data/rr_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# BP
frequencies_sys, bin_edges_sys = np.histogram(chunks[(chunks['frame_avg_bp_sys'] >= 80)]['frame_avg_bp_sys'], bins=10)
frequencies_dia, bin_edges_dia = np.histogram(chunks[(chunks['frame_avg_bp_dia'] >= 40)]['frame_avg_bp_dia'], bins=10)
np.savetxt('data/bp_sys_histogram.csv', np.column_stack(((bin_edges_sys[:-1]+bin_edges_sys[1:])/2, frequencies_sys)), delimiter=',', header='BinEdges,Frequency', comments='')
np.savetxt('data/bp_dia_histogram.csv', np.column_stack(((bin_edges_dia[:-1]+bin_edges_dia[1:])/2, frequencies_dia)), delimiter=',', header='BinEdges,Frequency', comments='')

# SpO2
frequencies, bin_edges = np.histogram(chunks[(chunks['frame_avg_spo2'] >= 50)]['frame_avg_spo2'], bins=10)
np.savetxt('data/spo2_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

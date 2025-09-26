import os
import pandas as pd

data_dir = "Data/raw_data/annotated_gaze_data"
participant_times = {}

# Loop through participant folders
for pid_folder in os.listdir(data_dir):
    pid_path = os.path.join(data_dir, pid_folder)
    if not os.path.isdir(pid_path):
        continue

    gaze_dir = os.path.join(pid_path, "annotated_gaze")
    if not os.path.exists(gaze_dir):
        continue

    for file in os.listdir(gaze_dir):
        if file.endswith(".csv") and "gaze_writing" in file:
            filepath = os.path.join(gaze_dir, file)
            df = pd.read_csv(filepath, low_memory=False)

            if df.empty:
                continue

            # Get first and last system timestamps
            first_timestamp = df['system_timestamp'].iloc[0]
            last_timestamp = df['system_timestamp'].iloc[-1]

            duration_sec = (last_timestamp - first_timestamp) / 1_000_000
            pid = int(pid_folder)
            print(f"Total duration for file {file}: {duration_sec} seconds")
            participant_times[pid] = participant_times.get(pid, 0) + duration_sec

# Convert to hours
participant_times_hours = {pid: t/3600 for pid, t in participant_times.items()}
total_hours = sum(participant_times_hours.values())

print("Writing eye-tracking time per participant (hours):")
for pid, hours in sorted(participant_times_hours.items()):
    print(f"Participant {pid}: {hours:.2f} h")

print(f"\nTotal writing eye-tracking time: {total_hours:.2f} h")

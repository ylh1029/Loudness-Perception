'''
***************************
CREATE VIRTUAL ENVIRONMENT
***************************
Windows:
python -m venv .venv

Mac:
python3 -m venv .venv

***************************
ACTIVATE VIRTUAL ENVIRONMENT
***************************
Windows - Command Prompt:
.venv\Scripts\activate.bat

Windows - Powershell:
.venv\Scripts\Activate.ps1

Mac - Terminal:
source .venv/bin/activate

NOTE: When it works, you should see (.venv) at the beginning of your terminal prompt.

***************************
INSTALL DESIRED PACKAGES
***************************
Windows:
python -m pip install {insert list of libraries/packages}

Mac:
python3 -m pip install {insert list of libraries/packages}

Example:
python3 -m pip install numpy pandas matplotlib mne pathlib

NOTE: If prompted, feel free to upgrade pip using the following command
pip install --upgrade pip

***************************
SAVE PACKAGE LIST (OPTIONAL)
***************************
Windows:
python -m pip freeze > requirements.txt

Mac:
python3 -m pip freeze > requirements.txt

***************************
DOWNLOAD REQUIREMENTS.TXT
***************************
Windows:
python -m pip install -r requirements.txt

Mac:
python3 -m pip install -r requirements.txt

NOTE: Make sure the requirements.txt file is in same directory

***************************
VERIFY INSTALLED PACKAGES
***************************
Windows:
python -m pip list

Mac:
python3 -m pip list

***************************
***************************
'''
"""A small first-pass EDA pipeline for the BIDS EEG data in sub-01."""

from pathlib import Path
import json

import mne
import pandas as pd


# File guide: .vhdr/.vmrk/.eeg are the raw BrainVision triplet; MNE opens the
# .vhdr. The *_eeg.json, *_channels.tsv, and *_events.tsv files describe it.
# Resolve paths from this script so it works from any current working directory.
DATASET_DIR = Path(__file__).resolve().parent / "dummy_data" / "ds007615"
EEG_DIR = DATASET_DIR / "sub-01" / "eeg"

# Choose which recording to plot when raw files are available. Set to None to use
# the first recording with complete BrainVision files.
PLOT_RECORDING = "sub-01_task-rest_acq-ec"


def load_sidecars(eeg_json_path):
    """Read one recording's JSON metadata, channel table, and event table."""
    recording_name = eeg_json_path.name.removesuffix("_eeg.json")
    metadata = json.loads(eeg_json_path.read_text(encoding="utf-8"))

    # BIDS sidecars share a filename stem with the EEG recording.
    channels_path = EEG_DIR / f"{recording_name}_channels.tsv"
    events_path = EEG_DIR / f"{recording_name}_events.tsv"

    # utf-8-sig also handles the optional byte-order mark found in TSV files.
    channels = pd.read_csv(channels_path, sep="\t", encoding="utf-8-sig")
    events = pd.read_csv(events_path, sep="\t", encoding="utf-8-sig")

    return recording_name, metadata, channels, events


def inspect_metadata():
    """Summarize each recording and return its sidecar tables."""
    if not EEG_DIR.is_dir():
        raise FileNotFoundError(f"EEG folder not found: {EEG_DIR}")

    eeg_json_files = sorted(EEG_DIR.glob("*_eeg.json"))
    if not eeg_json_files:
        raise FileNotFoundError(f"No EEG JSON sidecars found in {EEG_DIR}")

    recordings = {}
    summary_rows = []

    for eeg_json_path in eeg_json_files:
        name, metadata, channels, events = load_sidecars(eeg_json_path)
        recordings[name] = {"metadata": metadata, "channels": channels, "events": events}

        # BrainVision is a three-file format: header, marker text, and binary signal.
        triplet = [
            EEG_DIR / f"{name}_eeg.vhdr",
            EEG_DIR / f"{name}_eeg.vmrk",
            EEG_DIR / f"{name}_eeg.eeg",
        ]
        missing_files = [path.name for path in triplet if not path.is_file()]

        channel_counts = channels["type"].value_counts().to_dict()
        summary_rows.append(
            {
                "recording": name,
                "task": metadata.get("TaskName", "unknown"),
                "duration_s": metadata.get("RecordingDuration"),
                "sampling_Hz": metadata.get("SamplingFrequency"),
                "channels": ", ".join(f"{kind}={count}" for kind, count in channel_counts.items()),
                "events": len(events),
                "raw_files_ready": not missing_files,
                "missing_raw_files": ", ".join(missing_files),
            }
        )

    summary = pd.DataFrame(summary_rows)
    print("\nRecording overview")
    print(summary.to_string(index=False))

    # Event labels tell us what the timestamps in each events.tsv represent.
    print("\nEvent counts by recording")
    for name, data in recordings.items():
        print(f"\n{name}")
        print(data["events"]["trial_type"].value_counts().to_string())

    # The channel table gives channel names, EEG/EOG type, units, and status.
    first_name = next(iter(recordings))
    print(f"\nChannel types for {first_name}")
    print(recordings[first_name]["channels"]["type"].value_counts().to_string())
    print("Channel names:", ", ".join(recordings[first_name]["channels"]["name"]))

    return recordings, summary


def load_raw_recordings(recordings):
    """Load available BrainVision recordings as lazy MNE Raw objects."""
    raw_by_name = {}

    for name in recordings:
        vhdr_path = EEG_DIR / f"{name}_eeg.vhdr"
        vmrk_path = EEG_DIR / f"{name}_eeg.vmrk"
        eeg_path = EEG_DIR / f"{name}_eeg.eeg"

        # Do not ask MNE to open an incomplete triplet (common with annex pointers).
        if not all(path.is_file() for path in (vhdr_path, vmrk_path, eeg_path)):
            continue

        # MNE reads the .vhdr entry point and follows references to the .vmrk/.eeg files.
        raw_by_name[name] = mne.io.read_raw_brainvision(
            vhdr_path,
            eog=("HOG1", "HOG2", "VOG1", "VOG2"),
            preload=False,
            verbose="ERROR",
        )

    return raw_by_name


def inspect_raw(raw_by_name):
    """Print a basic signal summary and open two standard MNE views."""
    if not raw_by_name:
        print(
            "\nRaw EEG signals are unavailable: this folder has broken Git-annex "
            "pointers for the .vhdr/.vmrk/.eeg files. The JSON, channels.tsv, and "
            "events.tsv metadata were still explored above. Restore/download the "
            "BrainVision triplets to run signal plots and summaries."
        )
        return

    # Use the selected recording if present, otherwise the first loadable one.
    recording_name = PLOT_RECORDING if PLOT_RECORDING in raw_by_name else next(iter(raw_by_name))
    raw = raw_by_name[recording_name]

    print(f"\nMNE Raw overview: {recording_name}")
    print(raw)
    print("Sampling rate (Hz):", raw.info["sfreq"])
    print("Duration (s):", round(raw.n_times / raw.info["sfreq"], 2))
    print("Channels:", len(raw.ch_names), "|", ", ".join(raw.ch_names))
    print("Annotations/events:", len(raw.annotations))

    # Summarize the first ten seconds of EEG; MNE stores EEG voltages in volts.
    eeg_names = [
        channel_name
        for channel_name, channel_type in zip(raw.ch_names, raw.get_channel_types())
        if channel_type == "eeg"
    ]
    stop_sample = min(raw.n_times, int(10 * raw.info["sfreq"]))
    eeg_data_volts = raw.get_data(picks=eeg_names, start=0, stop=stop_sample)
    eeg_data_microvolts = eeg_data_volts * 1e6
    signal_summary = pd.DataFrame(
        {
            "channel": eeg_names,
            "mean_uV": eeg_data_microvolts.mean(axis=1),
            "std_uV": eeg_data_microvolts.std(axis=1),
            "peak_to_peak_uV": eeg_data_microvolts.max(axis=1) - eeg_data_microvolts.min(axis=1),
        }
    )
    print("\nFirst 10 seconds of EEG (microvolts; descriptive only)")
    print(signal_summary.round(2).to_string(index=False))

    # Browse a short multichannel time window, then inspect its frequency content.
    raw.plot(n_channels=min(20, len(raw.ch_names)), duration=10, scalings="auto")
    raw.compute_psd(picks="eeg", fmax=60).plot()


if __name__ == "__main__":
    recording_data, _ = inspect_metadata()
    loaded_raw = load_raw_recordings(recording_data)
    inspect_raw(loaded_raw)

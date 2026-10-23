r'''
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
"""A small first-pass EDA pipeline for one BIDS EEG participant."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


# File guide: .vhdr/.vmrk/.eeg are the raw BrainVision triplet; MNE opens the
# .vhdr. The *_eeg.json, *_channels.tsv, and *_events.tsv files describe it.
# Resolve paths from this script so it works from any current working directory.
DATASET_DIR = Path(__file__).resolve().parent / "dummy_data" / "ds007615"
SUBJECT_ID = "sub-03"  # Change this to another subject folder, e.g. "sub-01".
EEG_DIR = DATASET_DIR / SUBJECT_ID / "eeg"
OUTPUT_DIR = DATASET_DIR / "eda_output" / SUBJECT_ID

# Set a recording stem here, or leave None to use the first complete recording.
PLOT_RECORDING = None


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


def plot_metadata(recordings):
    """Plot event timing and channel inventory from the available sidecars."""
    ldaep_name = next(name for name in recordings if "task-ldaep" in name)
    ldaep_events = recordings[ldaep_name]["events"]
    ldaep_labels = sorted(
        ldaep_events["trial_type"].unique(),
        key=lambda label: int(label.rsplit("_", 1)[1]),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Each dot is a stimulus marker from events.tsv, not a measured EEG value.
    timeline_ax = axes[0, 0]
    for y_position, label in enumerate(ldaep_labels):
        matching_events = ldaep_events[ldaep_events["trial_type"] == label]
        timeline_ax.scatter(
            matching_events["onset"] / 60,
            [y_position] * len(matching_events),
            s=12,
            label=label,
        )
    timeline_ax.set_yticks(range(len(ldaep_labels)), ldaep_labels)
    timeline_ax.set_xlabel("Minutes from recording start")
    timeline_ax.set_title("LDAEP stimulus event timeline")
    timeline_ax.grid(axis="x", alpha=0.25)

    # Count how many event markers were recorded for each sound level.
    count_ax = axes[0, 1]
    event_counts = ldaep_events["trial_type"].value_counts().reindex(ldaep_labels)
    count_ax.bar(ldaep_labels, event_counts.values)
    count_ax.set_xlabel("LDAEP event type / sound level")
    count_ax.set_ylabel("Number of markers")
    count_ax.set_title("LDAEP event counts")

    # Rest markers show task start/end timing; they do not contain signal samples.
    rest_ax = axes[1, 0]
    rest_rows = []
    for name, data in recordings.items():
        if "task-rest" not in name:
            continue
        condition = "eyes closed" if "acq-ec" in name else "eyes open"
        for _, event in data["events"].iterrows():
            marker = str(event["trial_type"]).rsplit("_", 1)[-1]
            rest_rows.append((f"{condition}: {marker}", event["onset"]))

    if rest_rows:
        rest_labels = [
            "eyes closed: start",
            "eyes closed: end",
            "eyes open: start",
            "eyes open: end",
        ]
        rest_y = {label: index for index, label in enumerate(rest_labels)}
        for label, onset in rest_rows:
            color = "tab:green" if label.endswith("start") else "tab:red"
            rest_ax.scatter(onset, rest_y[label], color=color, s=45)
        rest_ax.set_yticks(range(len(rest_labels)), rest_labels)
        rest_ax.set_xlabel("Seconds from recording start")
        rest_ax.set_title("Resting-state start/end markers")
        rest_ax.grid(axis="x", alpha=0.25)
    else:
        rest_ax.axis("off")
        rest_ax.text(0.5, 0.5, "No resting-state event file for this subject", ha="center")

    # Channel counts come from channels.tsv; this is not a sensor-position map.
    channels = recordings[ldaep_name]["channels"]
    channel_counts = channels["type"].value_counts().sort_index()
    channel_ax = axes[1, 1]
    channel_ax.bar(channel_counts.index, channel_counts.values)
    channel_ax.set_ylabel("Number of channels")
    channel_ax.set_title("Channel types")

    fig.suptitle("Available EEG metadata (no EEG voltage data plotted)")
    fig.tight_layout()

    # Save a copy so the figure is available after the script exits.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "metadata_eda.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"\nSaved metadata visualization to: {output_path}")
    plt.close(fig)


def inspect_raw(raw_by_name, recordings):
    """Summarize real signal data and save a trace, PSD, and ERP figure."""
    if not raw_by_name:
        print(
            "\nNo complete BrainVision triplet was found. Metadata plots were still saved."
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
    print("\nFirst 10 seconds of raw EEG (µV; means include DC offsets, descriptive only)")
    print(signal_summary.round(2).to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter a copy for an easier-to-read trace; the original Raw data stay unchanged.
    trace_channels = [name for name in ("Fz", "Cz", "Pz") if name in raw.ch_names]
    trace_raw = raw.copy().pick(trace_channels).load_data()
    trace_raw.filter(l_freq=0.5, h_freq=30.0, verbose="ERROR")
    trace_data_uv = trace_raw.get_data() * 1e6
    trace_stop = min(trace_raw.n_times, int(10 * trace_raw.info["sfreq"]))

    # Welch PSD on the unfiltered Cz signal can reveal broad spectral shape and 50-Hz noise.
    spectrum = raw.compute_psd(
        method="welch",
        fmin=0.5,
        fmax=100,
        picks=["Cz"] if "Cz" in raw.ch_names else [trace_channels[0]],
        n_fft=int(2 * raw.info["sfreq"]),
        verbose="ERROR",
    )
    psd, frequencies = spectrum.get_data(return_freqs=True)

    # Use event.tsv sample indices to cut one-second EEG trials around each tone.
    ldaep_name = next(name for name in recordings if "task-ldaep" in name)
    events_table = recordings[ldaep_name]["events"]
    event_id = events_table.groupby("trial_type")["value"].first().astype(int).to_dict()
    events_array = np.column_stack(
        (
            events_table["sample"].astype(int).to_numpy() + raw.first_samp,
            np.zeros(len(events_table), dtype=int),
            events_table["value"].astype(int).to_numpy(),
        )
    )

    # This light 0.5-30 Hz filter and pre-tone baseline are for visualization only.
    erp_channel = "Cz" if "Cz" in raw.ch_names else trace_channels[0]
    erp_raw = raw.copy().pick([erp_channel]).load_data()
    erp_raw.filter(l_freq=0.5, h_freq=30.0, verbose="ERROR")
    epochs = mne.Epochs(
        erp_raw,
        events_array,
        event_id=event_id,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0),
        preload=True,
        verbose="ERROR",
    )
    print("\nLDAEP epochs extracted by sound level (no amplitude-based artifact rejection)")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Plot a short real EEG segment after filtering to remove slow drift and high-frequency noise.
    times = trace_raw.times[:trace_stop]
    for channel_index, channel_name in enumerate(trace_channels):
        axes[0].plot(times, trace_data_uv[channel_index, :trace_stop], label=channel_name, linewidth=0.8)
    axes[0].set_title("EEG time series: first 10 seconds, 0.5–30 Hz filtered")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Voltage (µV)")
    axes[0].legend(ncol=len(trace_channels))

    # Display PSD in dB/Hz and mark the dataset's 50-Hz power-line frequency.
    axes[1].plot(frequencies, 10 * np.log10(np.maximum(psd[0], np.finfo(float).tiny)))
    axes[1].axvline(50, color="tab:red", linestyle="--", label="50 Hz power line")
    axes[1].set_title("Cz power spectral density (unfiltered recording)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (dB/Hz)")
    axes[1].legend()

    # Average the tone-locked epochs separately for each sound level.
    for condition in sorted(event_id, key=lambda label: int(label.rsplit("_", 1)[1])):
        condition_epochs = epochs[condition]
        evoked = condition_epochs.average()
        axes[2].plot(
            evoked.times * 1000,
            evoked.data[0] * 1e6,
            label=f"{condition} (n={len(condition_epochs)})",
        )
        print(f"{condition}: {len(condition_epochs)} epochs")
    axes[2].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_title(f"Average tone-locked response at {erp_channel}")
    axes[2].set_xlabel("Time from tone onset (ms)")
    axes[2].set_ylabel("Baseline-corrected voltage (µV)")
    axes[2].legend()

    fig.suptitle(f"{SUBJECT_ID}: first-pass LDAEP signal EDA")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = OUTPUT_DIR / "signal_eda.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved signal visualization to: {output_path}")


if __name__ == "__main__":
    recording_data, _ = inspect_metadata()
    plot_metadata(recording_data)
    loaded_raw = load_raw_recordings(recording_data)
    inspect_raw(loaded_raw, recording_data)

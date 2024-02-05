import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec

# set matplotlib font to times new roman
matplotlib.rcParams["font.family"] = "Times New Roman"
DEFAULT_FONT_SIZE = 9

DATAPATH = "../Evaluation Data.xlsx"
DPI = 600
COLORS = ["#f2ae76", "#e34e64", "#882782"]

def f1(font_size=DEFAULT_FONT_SIZE):
    matplotlib.rcParams.update({"font.size": font_size})
    START = 3.25  # sec
    DUR = 3  # sec
    orig, sr = librosa.load("1 original.wav", offset=START, duration=DUR)
    afterHP = librosa.load("1 after highpass.wav", offset=START, duration=DUR)[0]
    afterNR = librosa.load("1 after noise reduction.wav", offset=START, duration=DUR)[0]
    afterNorm = librosa.load("1 after normalization.wav", offset=START, duration=DUR)[0]

    origDb = librosa.amplitude_to_db(np.abs(librosa.stft(orig)))
    afterHPDb = librosa.amplitude_to_db(np.abs(librosa.stft(afterHP)))
    afterNRDb = librosa.amplitude_to_db(np.abs(librosa.stft(afterNR)))
    afterNormDb = librosa.amplitude_to_db(np.abs(librosa.stft(afterNorm)))

    # Plot the spectrogram
    plt.figure(figsize=(8, 5.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

    plt.subplot(gs[0, 0])
    plt.title("(a) Original")
    librosa.display.specshow(
        origDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=origDb.min(),
        vmax=origDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(gs[0, 1], sharey=plt.gca())
    plt.title("(b) After High Pass")
    librosa.display.specshow(
        afterHPDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=origDb.min(),
        vmax=origDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(gs[1, 0], sharey=plt.gca())
    plt.title("(c) After Normalization")
    librosa.display.specshow(
        afterNormDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=origDb.min(),
        vmax=origDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(gs[1, 1], sharey=plt.gca())
    plt.title("(d) After Noise Reduction")
    librosa.display.specshow(
        afterNRDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=origDb.min(),
        vmax=origDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    # Create colorbar using the last column of the gridspec
    cbar = plt.subplot(gs[:, 2])
    plt.colorbar(format="%+2.0f dB", cax=cbar)

    plt.tight_layout()

    # Save to png
    plt.savefig("1.png", dpi=DPI)


def f3(font_size=DEFAULT_FONT_SIZE):
    matplotlib.rcParams.update({"font.size": font_size})
    START = 0.5  # sec
    DUR = 3  # sec
    before, sr = librosa.load("3 before.wav", offset=START, duration=DUR)
    after = librosa.load("3 after.wav", offset=START, duration=DUR)[0]

    START = 68.4  # sec
    DUR = 3  # sec
    bestRaw = librosa.load("3 best raw.wav", offset=START, duration=DUR)[0]

    START = 0.3  # sec
    DUR = 3  # sec
    forest, sr = librosa.load("3 forest after.wav", offset=START, duration=DUR)
    noForest = librosa.load("3 no forest after.wav", offset=START, duration=DUR)[0]

    beforeDb = librosa.amplitude_to_db(np.abs(librosa.stft(before)))
    afterDb = librosa.amplitude_to_db(np.abs(librosa.stft(after)))
    bestRawDb = librosa.amplitude_to_db(np.abs(librosa.stft(bestRaw)))
    forestDb = librosa.amplitude_to_db(np.abs(librosa.stft(forest)))
    noForestDb = librosa.amplitude_to_db(np.abs(librosa.stft(noForest)))

    # Plot the spectrogram
    plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.05])

    plt.subplot(gs[0, 0])
    plt.title("(a) No Forest")
    librosa.display.specshow(
        noForestDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=forestDb.min(),
        vmax=forestDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(gs[0, 1], sharey=plt.gca())
    plt.title("(b) Forest")
    librosa.display.specshow(
        forestDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=forestDb.min(),
        vmax=forestDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(gs[1, 0], sharey=plt.gca())
    plt.title("(c) Before DSSS")
    librosa.display.specshow(
        beforeDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=beforeDb.min(),
        vmax=beforeDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(gs[1, 1], sharey=plt.gca())
    plt.title("(d) After DSSS")
    librosa.display.specshow(
        afterDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=beforeDb.min(),
        vmax=beforeDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(gs[2, 0], sharey=plt.gca())
    plt.title("(e) Best Raw Recording")
    librosa.display.specshow(
        bestRawDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=beforeDb.min(),
        vmax=beforeDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    # Create colorbar using the last column of the gridspec
    cbar = plt.subplot(gs[:, 2])
    plt.colorbar(format="%+2.0f dB", cax=cbar)

    plt.tight_layout()

    # Save to png
    plt.savefig("3.png", dpi=DPI)


def f4(font_size=DEFAULT_FONT_SIZE):
    matplotlib.rcParams.update({"font.size": font_size})
    asong, sr = librosa.load("4 a song after.wav", offset=3, duration=3)
    bsong = librosa.load("4 b song after.wav", offset=4.5, duration=3)[0]

    asongDb = librosa.amplitude_to_db(np.abs(librosa.stft(asong)))
    bsongDb = librosa.amplitude_to_db(np.abs(librosa.stft(bsong)))

    # Plot the spectrogram
    plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    plt.subplot(gs[0, 0])
    plt.title("(a) A Song")
    librosa.display.specshow(
        asongDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=asongDb.min(),
        vmax=asongDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(gs[0, 1], sharey=plt.gca())
    plt.title("(b) B Song")
    librosa.display.specshow(
        bsongDb,
        sr=sr,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        vmin=asongDb.min(),
        vmax=asongDb.max(),
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    # Create colorbar using the last column of the gridspec
    cbar = plt.subplot(gs[:, 2])
    plt.colorbar(format="%+2.0f dB", cax=cbar)

    plt.tight_layout()

    # Save to png
    plt.savefig("4.png", dpi=DPI)


def make_bar(
    df: pd.DataFrame, title: str, xlabel: str, ylabel: str, bar_width: float = 0.25
):
    base_positions = np.arange(df.shape[0])
    bar_positions = [base_positions + i * bar_width for i in range(df.shape[1] - 1)]

    plt.grid(axis="y", which="major", linewidth=0.5, alpha=0.7, zorder=0)

    for i, col in enumerate(df.columns[1:]):
        plt.bar(
            bar_positions[i],
            df[col],
            width=bar_width,
            label=col,
            zorder=5,
            color=COLORS[i],
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 50)

    plt.legend(loc="lower right")

    plt.xticks(bar_positions[0] + bar_width, df.iloc[:, 0])


def make_line(
    df: pd.DataFrame, title: str, xlabel: str, ylabel: str, bar_width: float = 0.25
):
    plt.grid(axis="y", which="major", linewidth=0.5, alpha=0.7, zorder=0)

    for i, col in enumerate(df.columns[1:]):
        plt.plot(
            df[df.columns[0]],
            df[col],
            label=col,
            marker="o",
            markersize=3.5,
            zorder=5,
            color=COLORS[i],
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 50)

    plt.legend(loc="lower right")


def f2(font_size=DEFAULT_FONT_SIZE):
    matplotlib.rcParams.update({"font.size": font_size})

    df = pd.read_excel(DATAPATH, sheet_name="All Other Tests")
    df.columns = [x.strip() for x in df.columns]
    df = df[
        [
            "Graph: Test",
            "Graph: Label",
            "Graph: Average SDR",
            "Graph: Average SIR",
            "Graph: Average SAR",
        ]
    ]
    df.columns = [x.replace("Graph: ", "") for x in df.columns]
    table_starts = df[~pd.isna(df["Test"])].index.values
    table_ends = df[df["Label"] == "Grand Total"].index.values
    # insert table_starts[0]+2 at start of table_ends
    table_ends = np.insert(table_ends, 0, table_starts[0] + 2)

    tables = {}
    for start, end in zip(table_starts, table_ends):
        name = df.iloc[start]["Test"]
        table = df.iloc[start + 1 : end, 1:]
        tables[name] = table

    # keys = ['Control', 'Song', 'Double Background', 'Forest', 'Num Pos Songs', 'IBR']

    plt.figure(figsize=(8, 8))

    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])

    plt.subplot(gs[0, 0])
    df = pd.read_excel(DATAPATH, sheet_name="Model Comp")
    df.columns = [x.strip() for x in df.columns]
    make_bar(df, "(a) SDR/SIR/SAR vs Model Type", "Model Type", "SDR/SIR/SAR (dB)")
    plt.ylim(0, 40)

    plt.subplot(gs[0, 1])
    make_bar(
        tables["Song"], "(b) SDR/SIR/SAR vs Song Type", "Song Type", "SDR/SIR/SAR (dB)"
    )

    plt.subplot(gs[1, 0])
    make_line(
        tables["Num Pos Songs"],
        "(c) SDR/SIR/SAR vs Birdsongs per Minute",
        "Birdsongs per Minute",
        "SDR/SIR/SAR (dB)",
    )
    plt.xticks(range(0, 21, 2))

    plt.subplot(gs[1, 1])
    make_line(
        tables["IBR"],
        "(d) SDR/SIR/SAR vs Birdsong-Background Ratio",
        "Birdsong-Background Ratio",
        "SDR/SIR/SAR (dB)",
    )
    plt.xscale("log")
    plt.xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2])
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

    plt.subplot(gs[2, 0])
    make_bar(
        tables["Double Background"],
        "(e) SDR/SIR/SAR vs Background Type",
        "Background Type",
        "SDR/SIR/SAR (dB)",
    )

    plt.subplot(gs[2, 1])
    make_bar(
        tables["Forest"],
        "(f) SDR/SIR/SAR vs Forest Type",
        "Forest Impulse Respone (in Thousands of Trees)",
        "SDR/SIR/SAR (dB)",
    )

    plt.tight_layout()
    plt.savefig("2.png", dpi=DPI)


if __name__ == "__main__":
    # Spectrograms
    # f1()
    f3()
    # f4()

    # Graphs
    # f2(8)

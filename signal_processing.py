import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy.fft import fft
from scipy.stats import skew


def compute_features(signal):

    mean_val = float(np.mean(signal))
    rms_val = float(np.sqrt(np.mean(signal**2)))
    peak_val = float(np.max(np.abs(signal)))
    p2p_val = float(np.max(signal) - np.min(signal))
    variance_val = float(np.var(signal))
    std_val = float(np.std(signal))
    crest_factor = float(peak_val / rms_val) if rms_val != 0 else 0
    skewness_val = float(skew(signal))

    return {
        "mean": mean_val,
        "rms": rms_val,
        "peak": peak_val,
        "peak_to_peak": p2p_val,
        "variance": variance_val,
        "std_dev": std_val,
        "crest_factor": crest_factor,
        "skewness": skewness_val
    }


def compute_fft_plot(signal, sampling_rate):

    N = len(signal)

    # =========================
    # APPLY HANNING WINDOW
    # =========================
    window = np.hanning(N)
    windowed_signal = signal * window

    # =========================
    # FFT
    # =========================
    yf = fft(windowed_signal)
    xf = np.fft.fftfreq(N, 1/sampling_rate)

    amplitude = (2.0 / N) * np.abs(yf[:N//2])

    plt.figure(figsize=(8,4))
    plt.plot(xf[:N//2], amplitude)
    plt.title("FFT Spectrum (Hanning Window Applied)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return image_base64


def compute_time_plot(signal, sampling_rate):

    N = len(signal)
    t = np.arange(N) / sampling_rate

    plt.figure(figsize=(8,4))
    plt.plot(t, signal)
    plt.title("Time Domain Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return image_base64


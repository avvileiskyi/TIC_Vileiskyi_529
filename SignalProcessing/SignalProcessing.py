import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import fft

mean = 0
std_dev = 10
n = 500

random_signal = np.random.normal(mean, std_dev, n)

Fs = 1000
time = np.arange(n) / Fs

F_max = 9
w = F_max / (Fs / 2)

sos = signal.butter(3, w, 'low', output='sos')

filtered_signal = signal.sosfiltfilt(sos, random_signal)

def plot_signal(x, y, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    fig.savefig(f"../SignalProcessing/figures/{filename}.png", dpi=600)
    plt.close(fig)

plot_signal(
    time,
    filtered_signal,
    title="Сигнал з максимальною частотою F_max=9 Гц",
    xlabel="Час (секунди)",
    ylabel="Амплітуда сигналу",
    filename="filtered_signal"
)


spectrum = fft.fft(filtered_signal)
spectrum_magnitude = np.abs(fft.fftshift(spectrum))

freqs = fft.fftfreq(n, 1/Fs)
freqs_shifted = fft.fftshift(freqs)

# оскільки корисний сигнал після фільтрації містить частоти лише до 9 Гц,
# можна обмежити діапазон частот до +-200 Гц для кращого розуміння спектра
# можливо це не треба було робити, але графік дуже вузький без обмеження

# обмеження частот для відображення [-200, 200] Гц:
mask = (freqs_shifted >= -200) & (freqs_shifted <= 200)
freqs_limited = freqs_shifted[mask]
spectrum_limited = spectrum_magnitude[mask]

# побудова спектру з обмеженою шкалою частот
plot_signal(
    freqs_limited,
    spectrum_limited,
    title="Спектр сигналу з максимальною частотою F_max = 9 Гц",
    xlabel="Частота (Гц)",
    ylabel="Амплітуда спектру",
    filename="signal_spectrum"
)

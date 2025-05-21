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

discrete_signals = []

for Dt in [2, 4, 8, 16]:
    discrete_signal = np.zeros(n)
    for i in range(0, round(n / Dt)):
        if i * Dt < n:
            discrete_signal[i * Dt] = filtered_signal[i * Dt]
    discrete_signals.append(list(discrete_signal))

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(time, discrete_signals[s], linewidth=1)
        s += 1

fig.supxlabel("Час (секунди)", fontsize=14)
fig.supylabel("Амплітуда сигналу", fontsize=14)
fig.suptitle("Сигнали з кроком дискретизації Dt = (2, 4, 8, 16)", fontsize=14)
fig.savefig("../SignalProcessing/figures/discrete_signals.png", dpi=600)
plt.close(fig)

discrete_spectrums = []
for sig in discrete_signals:
    spectrum = fft.fft(sig)
    spectrum_magnitude = np.abs(fft.fftshift(spectrum))
    discrete_spectrums.append(list(spectrum_magnitude))

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(freqs_shifted, discrete_spectrums[s], linewidth=1)
        s += 1

fig.supxlabel("Частота (Гц)", fontsize=14)
fig.supylabel("Амплітуда спектру", fontsize=14)
fig.suptitle("Спектри сигналів з кроком дискретизації Dt = (2, 4, 8, 16)", fontsize=14)
fig.savefig("../SignalProcessing/figures/discrete_spectrums.png", dpi=600)
plt.close(fig)

discrete_restored_signals = []
w_filter = 16 / (Fs / 2)
sos_filter = signal.butter(3, w_filter, 'low', output='sos')

for sig in discrete_signals:
    restored_signal = signal.sosfiltfilt(sos_filter, sig)
    discrete_restored_signals.append(list(restored_signal))

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(time, discrete_restored_signals[s], linewidth=1)
        s += 1

fig.supxlabel("Час (секунди)", fontsize=14)
fig.supylabel("Амплітуда сигналу", fontsize=14)
fig.suptitle("Відновлені аналогові сигнали з кроком дискретизації Dt = (2, 4, 8, 16)", fontsize=14)
fig.savefig("../SignalProcessing/figures/restored_signals.png", dpi=600)
plt.close(fig)

dispersions = []
snrs = []

for restored in discrete_restored_signals:
    diff = np.array(restored) - filtered_signal
    signal_var = np.var(filtered_signal)
    diff_var = np.var(diff)
    dispersions.append(diff_var)
    snrs.append(signal_var / diff_var)

Dts = [2, 4, 8, 16]

# дисперсія
plt.figure(figsize=(21/2.54, 14/2.54))
plt.grid(True)
plt.plot(Dts, dispersions, marker='o', linewidth=1)
plt.xlabel("Крок дискретизації", fontsize=14)
plt.ylabel("Дисперсія", fontsize=14)
plt.title("Залежність дисперсії від кроку дискретизації", fontsize=14)
plt.savefig("../SignalProcessing/figures/dispersion_vs_dt.png", dpi=600)
plt.close()

# ССШ
plt.figure(figsize=(21/2.54, 14/2.54))
plt.grid(True)
plt.plot(Dts, snrs, marker='o', linewidth=1)
plt.xlabel("Крок дискретизації Dt", fontsize=14)
plt.ylabel("ССШ", fontsize=14)
plt.title("Залежність співвідношення сигнал-шум від кроку дискретизації", fontsize=14)
plt.savefig("../SignalProcessing/figures/snr_vs_dt.png", dpi=600)
plt.close()

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

quantized_signals = []
dispersions = []
snrs = []

quantize_tables_all = []
bit_sequences_all = []
M_values = [4, 16, 64, 256]

for M in M_values:
    bits = []
    bits_signal = []

    delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)

    quantize_signal = delta * np.round(filtered_signal / delta)

    quantize_levels = np.arange(np.min(quantize_signal), np.max(quantize_signal) + delta, delta)

    quantize_bit = np.arange(0, M)

    num_bits = 3

    quantize_bit = [format(i, f'0{int(np.ceil(np.log2(M)))}b') for i in range(M)]
    quantize_bit_for_table = [format(i, '04b') for i in range(min(M, 16))]
    quantize_table = np.c_[quantize_levels[:16], quantize_bit_for_table]
    quantize_tables_all.append(quantize_table)

    for val in quantize_signal:
        for idx, level in enumerate(quantize_levels[:M]):
            if np.round(np.abs(val - level), 8) == 0:
                bits.append(quantize_bit[idx])
                break

    bits = [int(bit) for bit in list(''.join(bits))]
    bit_sequences_all.append(bits)

    quantized_signals.append(quantize_signal)

    diff = quantize_signal - filtered_signal
    dispersions.append(np.var(diff))

    signal_var = np.var(filtered_signal)
    noise_var = np.var(diff)
    snrs.append(signal_var / noise_var)

# Таблиця квантування
fig, ax = plt.subplots(figsize=(14/2.54, 48/2.54))
full_table = np.vstack(quantize_tables_all)
table = ax.table(cellText=full_table, colLabels=['Значення сигналу', 'Кодова послідовність'], loc='center')
table.set_fontsize(14)
table.scale(1, 2)
ax.axis('off')
fig.savefig("../SignalProcessing/figures/quantization_table_all_M.png", dpi=600)
plt.close(fig)

# Бітові послідовності — графік 2x2 з одними підписами
fig, axes = plt.subplots(2, 2, figsize=(21/2.54*2, 14/2.54*2), sharex=True, sharey=True)
for ax, bits in zip(axes.flatten(), bit_sequences_all):
    ax.step(np.arange(len(bits)), bits, linewidth=0.1)
fig.suptitle("Кодова послідовність сигналу при різній кількості рівнів квантування", fontsize=14)
fig.supxlabel("Біти", fontsize=14)
fig.supylabel("Амплітуда сигналу", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("../SignalProcessing/figures/bit_sequence_all_M.png", dpi=600)
plt.close(fig)

# Сигнали після квантування
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
for idx, qsig in enumerate(quantized_signals):
    ax.plot(time, qsig, label=f'M={M_values[idx]}')
ax.set_xlabel("Час (секунди)", fontsize=14)
ax.set_ylabel("Амплітуда сигналу", fontsize=14)
ax.set_title("Цифрові сигнали з рівнями квантування (4, 16, 64, 256)", fontsize=14)
ax.legend()
fig.savefig("../SignalProcessing/figures/quantized_signals.png", dpi=600)
plt.close(fig)

# Дисперсія від M
plt.figure(figsize=(21/2.54, 14/2.54))
plt.plot(M_values, dispersions, marker='o', linewidth=1)
plt.grid(True)
plt.xlabel("Кількість рівнів квантування", fontsize=14)
plt.ylabel("Дисперсія", fontsize=14)
plt.title("Залежність дисперсії від кількості рівнів квантування", fontsize=14)
plt.savefig("../SignalProcessing/figures/dispersion_vs_M.png", dpi=600)
plt.close()

# ССШ від M
plt.figure(figsize=(21/2.54, 14/2.54))
plt.plot(M_values, snrs, marker='o', linewidth=1)
plt.grid(True)
plt.xlabel("Кількість рівнів квантування", fontsize=14)
plt.ylabel("ССШ", fontsize=14)
plt.title("Залежність співвідношення сигнал-шум від кількості рівнів квантування", fontsize=14)
plt.savefig("../SignalProcessing/figures/snr_vs_M.png", dpi=600)
plt.close()

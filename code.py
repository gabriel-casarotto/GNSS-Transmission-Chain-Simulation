from matplotlib import use
use("WebAgg")
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from scipy.special import erfc

#================================================
#  Conversion integer => bits and bits => integer 
#================================================
def int_to_bits(n, bit_length):
    b = bin(n)[2:].zfill(bit_length)
    return np.array(list(map(int, b)))

def bits_to_int(bits):
    return int("".join(bits.astype(str)), 2)

#===============================
#     Modulation BPSK/QPSK
#===============================
def BPSK_or_QPSK(bits, method):

    if method == "BPSK":
        return 2*bits - 1

    # QPSK
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    bits = bits.reshape(-1, 2)
    mapping = {
        (0,0): 1+1j,
        (0,1): -1+1j,
        (1,1): -1-1j,
        (1,0): 1-1j
    }
    symbols = np.array([mapping[tuple(b)] for b in bits]) / np.sqrt(2)

    return symbols 


#===============================
#   Probability error
#===============================

def Q(x):
    """Q-function"""
    return 0.5 * erfc(x / np.sqrt(2))

def Pe_BPSK(EbN0_dB):
    """
    Symbol/bit error probability for BPSK
    EbN0_dB : Eb/N0 in dB (scalar or numpy array)
    """
    EbN0 = 10**(EbN0_dB / 10)
    return Q(np.sqrt(2 * EbN0))

def Pe_QPSK(EbN0_dB):
    """
    Symbol error probability for QPSK
    EbN0_dB : Eb/N0 in dB (scalar or numpy array)
    """
    EbN0 = 10**(EbN0_dB / 10)
    gamma = np.sqrt(2 * EbN0)
    return 2 * Q(gamma) - Q(gamma)**2

#===============================
#   Doppler effect
#===============================

def rx_chain_doppler_only(bits_message, method, fd_hz):
    """
    Chaîne TX -> RF (avec Doppler) -> downconversion -> LPF -> matched filter -> décimation
    SANS AWGN (aucun bruit).
    Retourne:
      - symbols : constellation idéale
      - rx_symb : constellation reçue avec rotation Doppler (sans dispersion bruit)
    """
    # Modulation (symboles idéaux)
    symbols = BPSK_or_QPSK(bits_message, method)

    # Pulse shaping rect + temps
    tx_bb = rect_pulse_shaping(symbols, sps)
    t = np.arange(len(tx_bb)) / Fs

    # Upconversion avec Doppler (porteuse Fc + fd)
    tx_rf = np.real(tx_bb * np.exp(1j * (2*np.pi*(Fc + fd_hz)*t + phi0)))

    # Downconversion (le récepteur mixe à Fc -> reste fd en BB => rotation)
    rx_bb = tx_rf * np.exp(-1j * 2 * np.pi * Fc * t)

    # LPF
    fc_lp = 1.2 * Rs
    Wn = fc_lp / (Fs/2)
    b, a = butter(6, Wn)
    rx_bb = filtfilt(b, a, rx_bb)

    # Matched filter rect (moyenne)
    mf = np.ones(sps)/sps
    rx_bb = np.convolve(rx_bb, mf[::-1], mode='same')

    # petit délai (comme ton code)
    delay = 100
    rx_bb = rx_bb[delay:]

    # Décimation symbole
    rx_symb = rx_bb[::sps]

    return symbols, rx_symb


#===============================
#   Pulse shaping function
#===============================
def rect_pulse_shaping(symbols, sps):
    return np.repeat(symbols, sps)

#===============================
#  Données GNSS (exemple)
#===============================
lat = 48.814556
lon = 2.396805
alt = 350
# timestamp = int(time.time())
# print(timestamp)
timestamp = 1766075651 #18/12/2025 5:33

lat_bits = int_to_bits(int(lat*1e7), 32)
lon_bits = int_to_bits(int(lon*1e7), 32)
alt_bits = int_to_bits(int(alt*100), 16)
time_bits = int_to_bits(timestamp, 32)

bits_message = np.concatenate([lat_bits, lon_bits, alt_bits, time_bits])
print("Total bits =", len(bits_message))

method = "BPSK"
symbols = BPSK_or_QPSK(bits_message, method)

#===============================
#  Affichage symboles
#===============================
if method == "QPSK":
    plt.figure()
    plt.stem(np.real(symbols), markerfmt='ro', label="I")
    plt.stem(np.imag(symbols), markerfmt='bo', label="Q")
    plt.grid(True)
    plt.title("Signal symboles (I/Q)")

    plt.figure()
    plt.plot(np.real(symbols), np.imag(symbols), "o")
    plt.title("Constellation QPSK")
    plt.grid(True)

if method == "BPSK":
    plt.figure()
    plt.stem(symbols, markerfmt='blue', label="I")
    plt.grid(True)
    plt.title("Signal symboles")

    plt.figure()
    plt.plot(np.real(symbols), np.imag(symbols), "o")
    plt.title("Constellation BPSK")
    plt.grid(True)

#===============================
# Pulse shaping RECT
#===============================
Rs = 1e6
sps = 300
Fs = Rs * sps #3 Mb/s

tx_bb = rect_pulse_shaping(symbols, sps)
t_bb = np.arange(len(tx_bb)) / Fs

plt.figure(figsize=(12,4))
plt.step(t_bb*1e6, np.real(tx_bb), label="I", where='post') # 1bit = 1microsecond
plt.step(t_bb*1e6, np.imag(tx_bb), label="Q", where='post')
plt.title("Signal baseband rectangulaire")
plt.xlabel("Temps (µs)")
plt.grid(True)
plt.legend()

#===============================
#  Spectrum
#===============================
N = len(tx_bb)
Xf = np.fft.fftshift(np.fft.fft(tx_bb))
f = np.fft.fftshift(np.fft.fftfreq(N, 1/Fs))

# Spectre en dB
Xf_dB = 20*np.log10(np.abs(Xf) + 1e-12)

# Niveau maximal du spectre
max_dB = np.max(Xf_dB)

# Seuil = max - 80 dB
threshold = max_dB - 80

# Masque : tout ce qui est sous -80 dB est coupé (mis à threshold)
Xf_cut_dB = np.maximum(Xf_dB, threshold)

plt.figure(figsize=(12,4))
plt.plot(f/1e6, Xf_cut_dB)
plt.xlabel("Fréquence (MHz)")
plt.ylabel("Amplitude (dB)")
plt.title(f"Spectrum at -80 dB - method {method}")
plt.grid(True)

#===============================
#  Up-conversion at 30 MHz
#===============================

# Frequency (separated from BB)
Fc = 30e6  # 50 MHz (>> Rs = 1 MHz)

# Base band complex signal
bx = tx_bb  # tx_bb is already complex (I + jQ)

# Temps
t_rf = t_bb

# Transmitted signal
tx_rf = np.real(bx * np.exp(1j * 2 * np.pi * Fc * t_rf))
I_t = np.real(tx_bb)
Q_t = np.imag(tx_bb)

# temporal plot
num_samples_display = int(6 * sps)
t_display = t_rf[:num_samples_display]
plt.figure(figsize=(12,4))
plt.plot(t_rf[:num_samples_display]*1e6, tx_rf[:num_samples_display])
plt.title(f"Signal transmis RF (réel) – {method} – Fc={Fc/1e6} MHz")
plt.xlabel("Temps (µs)")
plt.ylabel("Amplitude")
plt.grid(True)

#=====================================
#  Spectrum of transmitted signal (RF)
#=====================================
N_rf = len(tx_rf)
Xf_rf = np.fft.fftshift(np.fft.fft(tx_rf))
f_rf = np.fft.fftshift(np.fft.fftfreq(N_rf, 1/Fs))

# Spectrum in dB
Xf_rf_dB = 20*np.log10(np.abs(Xf_rf) + 1e-12)

# Max level of spectrum
max_rf_dB = np.max(Xf_rf_dB)

# Seuil = max - 80 dB
threshold_rf = max_rf_dB - 80

# Masque : tout ce qui est sous -80 dB est coupé (mis à threshold)
Xf_rf_cut_dB = np.maximum(Xf_rf_dB, threshold_rf)

plt.figure(figsize=(12,4))
plt.plot(f_rf/1e6, Xf_rf_cut_dB)
plt.xlabel("Fréquence (MHz)")
plt.ylabel("Amplitude (dB)")
plt.title(f"Spectrum RF at -80 dB - méthode {method} - Fc={Fc/1e6} MHz")
plt.grid(True)

#===============================
#  PSD comparison
#===============================
plt.figure(figsize=(12, 6))

# Baseband spectrum (referenced at 0 Hz)
plt.plot(f/1e6, Xf_cut_dB, label="Spectrum Base Band", linestyle='--')

# RF spectrum (centered on Fc and -Fc)
plt.plot(f_rf/1e6, Xf_rf_cut_dB, label="Spectrum RF")

plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude (dB)")
plt.title(f"Spectrum comparison (BB vs RF) - methode {method}")
plt.grid(True)
plt.legend()
plt.ylim(-10, 80)
plt.xlim(-1.5*Fc/1e6, 1.5*Fc/1e6) # To visualize two lobs

#===============================
#  Adding noise
#===============================
# noise parameter
signal_power = np.mean(tx_rf**2)
Eb = signal_power / Rs
EbN0_dB = 10
EbN0_dB_linear = 10**(EbN0_dB/10)
noise_power = Eb/EbN0_dB_linear
sigma = np.sqrt(noise_power*Fs/2)
#Noise generation
noise = sigma * np.random.randn(len(tx_rf))
#Signal with noise
rx_rf = tx_rf + noise

plt.figure(figsize=(12, 4))
plt.plot(t_display*1e6, rx_rf[:num_samples_display])
plt.title(f"Signal RF Reçu (with AWGN, EbN0_dB={EbN0_dB} dB and SNR_dB = {10*np.log10(np.mean(tx_rf**2) / np.mean(noise**2))})")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.figure()
plt.plot(rx_rf[:2000], label="rx_rf")
plt.plot(tx_rf[:2000], label="tx_rf", alpha=0.7)
plt.legend()
plt.title("tx_rf vs rx_rf")
plt.grid()

print("Signal power =", np.mean(tx_rf**2))
print("Noise power  =", np.mean(noise**2))
print("Measured SNR (dB) =", 10*np.log10(np.mean(tx_rf**2) / np.mean(noise**2)))

#===============================
#  Spectrum of noised signal
#===============================

N_rf_noised = len(rx_rf)
Xf_rf_noised = np.fft.fftshift(np.fft.fft(rx_rf))
f_rf_noised = np.fft.fftshift(np.fft.fftfreq(N_rf_noised, 1/Fs))

# Spectre en dB
Xf_rf_dB_noised = 20*np.log10(np.abs(Xf_rf_noised) + 1e-12)

# Max level of spectrum
max_rf_dB_noised = np.max(Xf_rf_dB_noised)

# Seuil = max - 80 dB
threshold_rf_noised = max_rf_dB_noised - 80

# Mask : everything under -80 dB is cut
Xf_rf_cut_dB_noised = np.maximum(Xf_rf_dB_noised, threshold_rf_noised)

plt.figure(figsize=(12,4))
plt.plot(f_rf_noised/1e6, Xf_rf_cut_dB_noised)
plt.xlabel("Fréquence (MHz)")
plt.ylabel("Amplitude (dB)")
plt.title(f"Spectrum RF at -80 dB - methode {method} - Fc={Fc/1e6} MHz")
plt.grid(True)

plt.figure(figsize=(12, 6))

# Spectre en bande de base (référence à 0 Hz)
# On normalise pour une meilleure comparaison
max_bb = np.max(Xf_rf_cut_dB_noised)
plt.plot(f_rf_noised/1e6, Xf_rf_cut_dB_noised - max_bb, label="Spectre noised (normalisé)", linestyle='--')

# Spectre RF (centré sur Fc et -Fc)
max_rf = np.max(Xf_rf_cut_dB)
plt.plot(f_rf/1e6, Xf_rf_cut_dB - max_rf, label="Spectre RF (normalisé)")

plt.xlabel("Fréquence (MHz)")
plt.ylabel("Amplitude (dB) relative au maximum")
plt.title(f"Comparaison des Spectres (BB vs RF) - méthode {method}")
plt.grid(True)
plt.legend()
plt.ylim(-90, 0) # Pour se concentrer sur la zone d'intérêt
plt.xlim(-1.5*Fc/1e6, 1.5*Fc/1e6) # Pour visualiser les deux lobes


#===============================
#  Down-conversion (I/Q)
#===============================
rx_bb = rx_rf * np.exp(-1j * 2 * np.pi * Fc * t_rf)

# Filtre passe-bas simple (moyenne glissante)
fc_lp = 1.2 * Rs
Wn = fc_lp / (Fs/2)
b, a = butter(6, Wn)

rx_bb = filtfilt(b,a,rx_bb)


# Filtre rectangulaire pour matched filter
mf = np.ones(sps)/sps 

# Application du matched filter (convolution)
rx_bb = np.convolve(rx_bb, mf[::-1], mode='same')


delay = 100
rx_bb = rx_bb[delay:]

# Décimation symbole
rx_symb = rx_bb[::sps]


plt.figure()
plt.plot(np.real(rx_bb[:3000]), label="I")
plt.plot(np.imag(rx_bb[:3000]), label="Q")
plt.legend(); plt.grid()
plt.title("BB après downconversion + LPF")


if method == "QPSK":
    plt.figure()
    plt.stem(np.real(rx_symb[:112]), markerfmt='ro', label="I")
    plt.stem(np.imag(rx_symb[:112]), markerfmt='bo', label="Q")
    plt.grid(True)
    plt.title("Signal symboles (I/Q)")

    plt.figure()
    plt.plot(np.real(rx_symb[:112]), np.imag(rx_symb[:112]), "o")
    plt.title("Constellation QPSK post demodulation")
    plt.grid(True)

if method == "BPSK":
    plt.figure()
    plt.stem(rx_symb[:112], markerfmt='blue', label="I")
    plt.grid(True)
    plt.title("Signal symboles")

    plt.figure()
    plt.plot(np.real(rx_symb[:112]), np.imag(rx_symb[:112]), "o")
    plt.title("Constellation BPSK post demodulation")
    plt.grid(True)

# #=================================
# Spectre du Signal Down-convertit
#===================================
N_rx = len(rx_bb)
Xf_rx = np.fft.fftshift(np.fft.fft(rx_bb))
f_rx = np.fft.fftshift(np.fft.fftfreq(N_rx, 1/Fs))

# Spectre en dB
Xf_rx_dB = 20 * np.log10(np.abs(Xf_rx) + 1e-12)

# Niveau maximal du spectre
max_rx_dB = np.max(Xf_rx_dB)

plt.figure(figsize=(12, 4))
plt.plot(f_rx / 1e6, Xf_rx_dB - max_rx_dB)
plt.xlabel("Fréquence (MHz)")
plt.ylabel("Amplitude (dB) relative au maximum")
plt.title(f"Spectre du Signal Down-converti (Bande de Base) - method {method}")
plt.grid(True)
plt.ylim(-90, 0)

#===================================
# Conversion symbols => information
#===================================

if method == "BPSK":
    rx_bits = (np.real(rx_symb) > 0).astype(int)

    #SER
    tx_symb = symbols[:len(rx_symb)]
    rx_symb_decided = np.where(np.real(rx_symb) > 0, 1, -1)
    Pe_BPSK_sim = np.mean(rx_symb_decided != tx_symb)
    print("SER BPSK (simulation) =", Pe_BPSK_sim)
    Pe_BPSK_th = Pe_BPSK(EbN0_dB)
    print("SER BPSK théorique =", Pe_BPSK_th)

if method == "QPSK":
    rx_bits = []

    #SER
    tx_symb = symbols[:len(rx_symb)]
    rx_symb_decided = np.sign(np.real(rx_symb)) + 1j*np.sign(np.imag(rx_symb))
    rx_symb_decided /= np.sqrt(2)
    Pe_QPSK_sim = np.mean(rx_symb_decided != tx_symb)
    print("SER QPSK (simulation) =", Pe_QPSK_sim)
    Pe_QPSK_th = Pe_QPSK(EbN0_dB)
    print("SER QPSK théorique =", Pe_QPSK_th)

    for s in rx_symb:
        I_bits = np.real(s)
        Q_bits = np.imag(s)

        if I_bits > 0 and Q_bits > 0:
            rx_bits += [0, 0]
        elif I_bits < 0 and Q_bits > 0:
            rx_bits += [0, 1]
        elif I_bits < 0 and Q_bits < 0:
            rx_bits += [1, 1]
        elif I_bits > 0 and Q_bits < 0:
            rx_bits += [1, 0]

    rx_bits = np.array(rx_bits)

latitude_final_bits = rx_bits[0:32]
longitude_final = rx_bits[32:64]
alt_bits_final  = rx_bits[64:80]
time_bits_final = rx_bits[80:112]

lat_final  = bits_to_int(latitude_final_bits)  / 1e7
lon_final  = bits_to_int(longitude_final)  / 1e7
alt_final  = bits_to_int(alt_bits_final)  / 100
time_final = bits_to_int(time_bits_final)

print("Latitude final  =", lat_final)
print("Longitude final =", lon_final)
print("Altitude final  =", alt_final)
print("Timestamp final =", time_final)

# ============================================================
# FIGURES DOPPLER ONLY (without noise) : 4 graphes
# - BPSK idéal vs BPSK avec Doppler
# - QPSK idéal vs QPSK avec Doppler
# ============================================================

# --- Réglages Doppler (pas de bruit)
fd = 5e3          # Hz (augmente à 10e3 ou 20e3 si rotation pas assez visible)
phi0 = 0.0        # rad

# --- Run BPSK (idéal + Doppler)
sym_bpsk, rx_bpsk_dopp = rx_chain_doppler_only(bits_message, "BPSK", fd)

# --- Run QPSK (idéal + Doppler)
sym_qpsk, rx_qpsk_dopp = rx_chain_doppler_only(bits_message, "QPSK", fd)

# Nombre de points affichés
Nshow_bpsk = min(600, len(sym_bpsk), len(rx_bpsk_dopp))
Nshow_qpsk = min(600, len(sym_qpsk), len(rx_qpsk_dopp))

# ====== 4 FIGURES ======

# (1) BPSK idéal
plt.figure()
plt.plot(np.real(sym_bpsk[:Nshow_bpsk]), np.imag(sym_bpsk[:Nshow_bpsk]), "o")
plt.grid(True)
plt.title("Constellation BPSK (ideal)")

# (2) BPSK avec Doppler (sans bruit)
plt.figure()
plt.plot(np.real(rx_bpsk_dopp[:Nshow_bpsk]), np.imag(rx_bpsk_dopp[:Nshow_bpsk]), "o")
plt.grid(True)
plt.title(f"Constellation BPSK (Doppler only, fd={fd/1e3:.1f} kHz)")

# (3) QPSK idéal
plt.figure()
plt.plot(np.real(sym_qpsk[:Nshow_qpsk]), np.imag(sym_qpsk[:Nshow_qpsk]), "o")
plt.grid(True)
plt.title("Constellation QPSK (ideal)")

# (4) QPSK avec Doppler (sans bruit)
plt.figure()
plt.plot(np.real(rx_qpsk_dopp[:Nshow_qpsk]), np.imag(rx_qpsk_dopp[:Nshow_qpsk]), "o")
plt.grid(True)
plt.title(f"Constellation QPSK (Doppler only, fd={fd/1e3:.1f} kHz)")

plt.show()

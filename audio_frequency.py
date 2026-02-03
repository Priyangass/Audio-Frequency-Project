import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# STEP 1: LOAD AUDIO FILES
# ===============================
audio1, sr = librosa.load("audio1.wav", sr=22050, mono=True)
audio2, _  = librosa.load("audio2.wav", sr=22050, mono=True)

# ===============================
# STEP 2: MAKE SAME LENGTH
# ===============================
min_len = min(len(audio1), len(audio2))
audio1 = audio1[:min_len]
audio2 = audio2[:min_len]

# ===============================
# STEP 3: NORMALIZE
# ===============================
audio1 = audio1 / np.max(np.abs(audio1))
audio2 = audio2 / np.max(np.abs(audio2))

# ===============================
# STEP 4: FFT (POWER SPECTRUM)
# ===============================
fft1 = np.abs(np.fft.rfft(audio1))**2
fft2 = np.abs(np.fft.rfft(audio2))**2

freqs = np.fft.rfftfreq(len(audio1), d=1/sr)

# ===============================
# STEP 5: SMOOTHING (VISUAL ONLY)
# ===============================
def smooth(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

fft1_smooth = smooth(fft1)
fft2_smooth = smooth(fft2)

# ===============================
# STEP 6: PLOT (MATCHES YOUR IMAGE)
# ===============================
# ===============================
# STEP 5: CLEAN & CLEAR PLOT
# ===============================
plt.figure(figsize=(10, 5))

plt.plot(freqs, fft1_smooth, label="Audio 1", linewidth=1)
plt.plot(freqs, fft2_smooth, label="Audio 2", linewidth=1)

plt.title("Frequency Domain Comparison of Two Audio Signals")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.xlim(0, 4000)      # 🔑 focus on audible speech range
plt.ylim(1e3, 1e8)     # 🔑 prevents huge scaling
plt.yscale("log")

plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ===============================
# STEP 7: FEATURE EXTRACTION
# ===============================
def extract_features(audio, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    rms = np.mean(librosa.feature.rms(y=audio))
    return np.hstack([mfcc, centroid, bandwidth, rolloff, rms])

features1 = extract_features(audio1, sr)
features2 = extract_features(audio2, sr)

# ===============================
# STEP 8: SIMILARITY
# ===============================
similarity_score = cosine_similarity(
    features1.reshape(1, -1),
    features2.reshape(1, -1)
)[0][0]

similarity_percentage = similarity_score * 100

# ===============================
# STEP 9: OUTPUT
# ===============================
print("===================================")
print("Audio Frequency Similarity Analysis")
print("===================================")
print(f"Similarity Score      : {similarity_score:.4f}")
print(f"Similarity Percentage : {similarity_percentage:.2f}%")

if similarity_percentage >= 90:
    print("Result: Very high similarity")
elif similarity_percentage >= 70:
    print("Result: Moderate similarity")
elif similarity_percentage >= 50:
    print("Result: Low similarity")
else:
    print("Result: Very different audio signals")

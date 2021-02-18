import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from scipy.io import wavfile
from scipy import linalg
from spectrum.levinson import LEVINSON
from spectrum.lpc import lpc
from scipy.fftpack import dct


class SpeechHelper:
    matplotlib.rc('font', size=25)

    HOP_LENGTH = 0.010
    PRE_PROCESSING_APPLIED = False

    def __init__(self, wave_path: str, frame_length: int = 32, ft_n: int = 512):
        # open the speech file
        self.rate, self.sig = wavfile.read(wave_path)

        self.final_sig = self.sig
        self.ft_sig = []

        self.frame_length = frame_length
        self.frame_sample = int((self.frame_length/1000) * self.rate)
        self.frame_step = int(self.HOP_LENGTH * self.rate)

        self.ft_n = ft_n
        self.window = signal.hamming(self.frame_sample)
        self.signal_name = wave_path.split("/")[1]

        # voiced signal
        self.v_sig = False

    def framing(self) -> np.ndarray:
        """
        Framing main signal.
        :return: Framed signal array
        """
        frames = []
        for i in range(1, self.sig.size, self.frame_step):
            frame = self.sig[i-1:i-1 + self.frame_sample]
            frame = np.pad(frame, (0, self.frame_sample - frame.size))
            frames.append(frame)

        return np.array(frames)

    @classmethod
    def pre_emphasis(cls, y: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis on each frame
        :param y: one of the main signal's frame
        :return: frame which we applied pre-emphasis on it
        """
        return signal.lfilter([1, -0.97], [1], y)

    def windowing(self, y: np.ndarray) -> np.ndarray:
        """
        Apply hamming window on each frame.
        :param y: one of the main signal's frame
        :return: frame which we applied hamming window on it
        """
        return y * self.window

    def pre_processing(self):
        """
        Apply all the pre-processing steps on the signal.
        First we calculate signal frames. Hyper-parameters are:
            self.frame_length, HOP_LENGTH, self.window which we define as class parameters. The values are from exercise.
        I check if pre-processing applied on this instance's signal by checking PRE_PROCESSING_APPLIED.
        :return: self.final_sig: Signal which is ready to use and pre-processing steps applied on that.
        """
        if self.PRE_PROCESSING_APPLIED:
            return

        framed_sig = self.framing()

        final_sig = []
        for y in framed_sig:
            y_pre_emphasis = self.pre_emphasis(y)
            y_final = self.windowing(y_pre_emphasis)
            final_sig.append(y_final)

        self.final_sig = np.array(final_sig)

        self.PRE_PROCESSING_APPLIED = True

    def frames_fft(self):
        """
        Calculate fourier transform on each signal's step and then calculate its magnitude
        then apply log transform on it and calculate dB (for the reasons that we discussed in the class earlier).
        :return: self.ft_sig: is signal's fourier transform's magnitude in dB.
        """
        signal_mag = np.abs(np.fft.fft(self.final_sig, n=self.ft_n))

        half = int((self.ft_n/2)+1)

        self.ft_sig = (20 * np.log10(signal_mag + 10e-12)).T[0:half, :]

    def plot_frame(self, frame_no: int):
        """
        Plot a frame of the signal
        :param frame_no: Signal's frame number
        :return: Plot the frame
        """
        self.pre_processing()
        frame = self.final_sig[frame_no, :]

        plt.figure(figsize=(20, 10))
        plt.plot(frame)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title('{} frame No. {}'.format(self.signal_name, frame_no))
        plt.savefig(fname="Outputs/{}_{}.png".format(self.signal_name, frame_no, format="PNG"))
        plt.show()

    def manual_spectrogram(self):
        """
        Plot spectrogram which we calculated manually.
        The signal that we use in this method is ft_sig (result of the self.frames_fft method.)
        :return: Manual spectrogram plot
        """
        delta_frq = self.rate / self.ft_n
        fq_fft = np.arange(0, (self.ft_n / 2) + 1) * delta_frq

        sig_dur = np.linspace(0, self.sig.size / self.rate, self.ft_sig.shape[1])

        f, ax = plt.subplots(figsize=(30, 10))
        ax.pcolormesh(sig_dur, fq_fft, self.ft_sig, cmap="gray", shading='auto')
        ax.axis('tight')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')
        ax.set_title('Manual Spectrogram for {}\n T= {}, fourier n= {}'.format(self.signal_name, self.frame_length, self.ft_n))
        plt.savefig(fname="Outputs/{} manual_spectrogram_{}_{}.png".format(self.signal_name, self.frame_length, self.ft_n), format="PNG")
        plt.show()

    def auto_spectrogram(self):
        """
        Plot spectrogram which calculated automatically using scipy library.
        We use raw signal in this method (self.sig)
        :return: automatic calculated spectrogram plot
        """
        freqs, times, Sx = signal.spectrogram(self.sig,
                                              fs=self.rate,
                                              window='hamming',
                                              nperseg=self.frame_sample,
                                              noverlap=self.frame_sample - self.frame_step,
                                              nfft=self.ft_n,
                                              scaling="spectrum")

        fig, ax = plt.subplots(figsize=(30, 10))
        ax.pcolormesh(times, freqs, 20 * np.log10(Sx), cmap="gray", shading='auto')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')
        ax.set_title('Auto Spectrogram for {}\n T= {}, fourier n= {}'.format(self.signal_name, self.frame_length, self.ft_n))
        plt.savefig(fname="Outputs/{} auto_spectrogram_{}_{}.png".format(self.signal_name, self.frame_length, self.ft_n), format="PNG")
        plt.show()

    def plot_all_spectrograms(self):
        """
        Plot all spectrograms which we want to compare in exercise.

        First we call pre_processing method. This method apply pre-processing steps on signal.
        Pre processing Steps are:
            1- framing
            2- pre-emphasis on each frame
            3- apply hamming window on each frame

        Then we call frames_fft method which calculates fft for each signal's frame.

        manual_spectrogram method plots the signal's spectrogram which we calculated manually.

        auto_spectrogram method plots the signal's spectrogram which is calculated automatically by scipy library.

        We plot all the spectrograms together and then we can compare them and then answer exercises.
        :return: Plots
        """

        self.pre_processing()

        # calculate each frame's fft
        self.frames_fft()

        # plot signal's spectrogram which we calculated manually
        self.manual_spectrogram()
        # plot signal's spectrogram which calculated automatically by scipy library
        self.auto_spectrogram()

    def plot_magnitude_spectrum(self, frame_no: int):
        """
        Plot magnitude spectrum of a specific frame number.
        :param frame_no: Number of the frame
        :return:
        """
        self.pre_processing()
        self.frames_fft()

        frame = self.ft_sig[:, frame_no]

        delta_frq = self.rate / self.ft_n
        fq_fft = np.arange(0, (self.ft_n/2)+1) * delta_frq

        plt.figure(figsize=(20, 10))
        plt.plot(fq_fft, frame)
        plt.title("{} Magnitude Spectrum frame No. {}\n T= {}, fourier n= {}".format(self.signal_name,
                                                                                            frame_no,
                                                                                            self.frame_length,
                                                                                            self.ft_n))

        plt.xlabel('Frequency')
        plt.ylabel('Magnitude (dB)')
        plt.savefig(fname="Outputs/{} magnitude_spectrum_{}_{}_{}.png".format(self.signal_name, frame_no,
                                                                              self.frame_length,
                                                                              self.ft_n), format="PNG")
        plt.show()

    @classmethod
    def auto_correlation(cls, v: np.array):
        """
        Calculate auto correlation of a vector using numpy correlate method
        :param v: Vector
        """
        auto_corr = np.correlate(v, v, 'full')
        auto_corr = auto_corr[auto_corr.size // 2:]

        return auto_corr / auto_corr.max()

    def lpc(self, frame_no: int, lpc_type="iac", p: int = 20) -> np.array:
        """
        -> Q. 4
        Calculate LPC coefficients. We calculate them in three ways (as in requested in exercise's questions)
        First way to calculate LPC coefficients is to use auto correlation matrix and solve it using inv(R) * r equation
         -> iac (Inverse auto correlation)
        Second way is to use levinson algorithm. we use levinson's method of spectrum library for this purpose
         -> levinson
        Third way is to use some builtin lpc methods. we use lpc's method of spectrum library -> builtin
        :param frame_no: Signal's frame number
        :param lpc_type: which of the three available ways to calculate LPC
        :param p: LPC coefficients count
        :return: LPC coefficients
        """
        assert lpc_type in ["iac", "levinson", "builtin"]

        self.pre_processing()
        frame = self.final_sig[frame_no, :]

        if lpc_type == "iac":
            auto_corr = self.auto_correlation(frame)
            return np.array(self.lpc_iac(auto_corr, p))
        elif lpc_type == "levinson":
            auto_corr = self.auto_correlation(frame)
            return np.array(self.lpc_levinson(auto_corr, p))
        else:
            return np.insert(np.array(self.lpc_bi(frame, p)[0]), 0, 1)

    @classmethod
    def lpc_iac(cls, auto_corr, p: int = 20) -> np.array:
        """
        Calculate LPC coefficients using inverse auto correlation
        :param auto_corr: Frame auto correlation
        :param p: LPC coefficients count
        :return: LPC coefficients
        """
        R = linalg.toeplitz(auto_corr[:p])
        r = auto_corr[1:p+1]

        return np.insert(linalg.inv(R).dot(-r), 0, 1)

    @classmethod
    def lpc_levinson(cls, auto_corr, p: int = 20) -> np.array:
        """
        Calculate LPC coefficients using levinson algorithm
        :param auto_corr: Frame auto correlation
        :param p: LPC coefficients count
        :return: LPC coefficients
        """
        a, _, _ = LEVINSON(auto_corr, p)
        return np.insert(a, 0, 1)

    @classmethod
    def lpc_bi(cls, frame: np.array, p: int = 20) -> list:
        """
        Calculate LPC coefficients using builtin function
        :param frame: Frame vector of signal
        :param p: LPC coefficients count
        :return:
        """
        return lpc(frame, p)

    def lpc_gain(self, frame_no: int, p: int = 20) -> np.array:
        """
        Calculate the Gain
        :param frame_no: Signal's frame number
        :param p: LPC coefficients count
        :return: Signal gain
        """
        return np.sqrt(self.lpc_bi(self.final_sig[frame_no, :], p)[1])

    def residual_signal(self, frame_no: int, p: int = 20, plot=True) -> np.array:
        """
        Calculate residual signal which is the convolution of LPC coefficients by the main signal
        :param frame_no: Signal's frame number
        :param p: LPC coefficients count
        :param plot: Whether to plot or return residual signal
        :return:
        """
        self.pre_processing()
        frame = self.final_sig[frame_no, :]

        lpc_coef = self.lpc(frame_no, "builtin", p)
        lpc_coef_padded = np.pad(lpc_coef, (0, frame.size - lpc_coef.size))

        residual = np.convolve(frame, lpc_coef_padded)

        if not plot:
            return residual[residual.nonzero()]

        plt.figure(figsize=(20, 10))
        plt.plot(residual[residual.nonzero()])
        plt.title("{} Residual signal of the frame No. {}".format(self.signal_name, frame_no))
        plt.savefig(fname="Outputs/{}_{} residual_signal.png".format(self.signal_name, frame_no, format="PNG"))
        plt.show()

    def plot_spectral_envelope(self, frame_no: int, p: int = 20, frame_spec_plot: bool = True, show_formants: int = 3):
        """
        Plot Spectral envelope of a specific frame
        :param frame_no: Signal's frame number
        :param p: LPC coefficients count
        :param frame_spec_plot: Whether Plot frame spectrum or not
        :param show_formants: Show formants in the plot or not
        :return: Plot frame's spectral envelope
        """
        lpc_coef = self.lpc(frame_no, "builtin", p)
        lpc_coef_padded = np.pad(lpc_coef, (0, self.final_sig[0].size - lpc_coef.size))

        A = np.fft.fft(lpc_coef_padded, n=self.ft_n)[0:int(self.ft_n//2)+1]
        A_mag = np.abs(A)
        H = 20 * np.log10(self.lpc_gain(frame_no) * (1 / A_mag) + 10e-12)

        delta_frq = self.rate / self.ft_n
        fq_fft = np.arange(0, (self.ft_n//2)+1) * delta_frq

        plt.figure(figsize=(20, 10))

        # find spectral envelope peaks
        peaks, _ = signal.find_peaks(H)

        pr = 1
        if frame_spec_plot:
            self.frames_fft()
            frame = self.ft_sig[:, frame_no]
            plt.plot(fq_fft, frame)

            pr = (frame / H).mean()

        plt.plot(fq_fft, pr * H)

        # plot formant lines
        plt.vlines(x=fq_fft[peaks[:show_formants]], ymin=0, ymax=pr * H[peaks[:show_formants]], color='red', linestyles="dashdot")

        for i, formant in enumerate(fq_fft[peaks[:show_formants]]):
            print("Formant {} frequency: {}".format(i+1, formant))

        plt.title("{} Spectral envelop of frame No. {}".format(self.signal_name, frame_no))
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude (dB)')
        plt.savefig(fname="Outputs/{}_{} spectral_envelop.png".format(self.signal_name, frame_no, format="PNG"))
        plt.show()

    def plot_frequency_response(self, signal_var, frame_no: int):
        """
        Plot frequency response of a signal. It calculate using freqz.
        :param signal_var: The signal
        :param frame_no: The frame number
        :return: Plot frequency response of a signal
        """
        point_count = (self.final_sig.shape[1] // 2)+1
        w, h = signal.freqz(signal_var, fs=self.rate, worN=point_count)

        delta_frq = self.rate / self.ft_n
        fq_fft = np.arange(0, point_count) * delta_frq

        plt.figure(figsize=(20, 10))
        plt.plot(fq_fft, -20 * np.log10(abs(h)))
        plt.title("{} Vocal Tract frequency response of frame No. {}".format(self.signal_name, frame_no))
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude (dB)')
        plt.savefig(fname="Outputs/{}_{} frequency_response.png".format(self.signal_name, frame_no, format="PNG"))
        plt.show()

    def short_time_energy(self, plot: bool = True):
        self.pre_processing()
        ste = 20 * np.log10((np.sum(self.final_sig ** 2, axis=1) / self.frame_sample) + 1e-12)

        if plot:
            plt.figure(figsize=(100, 10))
            plt.plot(ste)
            plt.title('Short-Time Energy')
            plt.ylabel('energy')
            plt.xlabel('frame')
            plt.show()

        return ste

    def vad(self, plot=False):
        ste = self.short_time_energy(plot=False)
        ste_tr_ind = np.where(ste >= 60)

        v_sig = self.final_sig[ste_tr_ind]
        v_sig_filtered = np.zeros(v_sig.shape)
        for i in range(v_sig.shape[0]):
            v_sig_filtered[i, :] = np.median(v_sig[min(0, i-2):max(i+3, v_sig.shape[0])], axis=0)

        self.v_sig = v_sig_filtered
        if plot:
            plt.figure(figsize=(100, 10))
            plt.plot(ste[ste_tr_ind])
            plt.title('Short-Time Energy')
            plt.ylabel('energy')
            plt.xlabel('frame')
            plt.show()

        return ste_tr_ind

    def filter_bank(self, filter_count: int = 20, start_freq: int = 150, end_freq: int = 3800, plot: bool = False):
        start_mel = 2595 * np.log10(1 + (start_freq / 700))
        end_mel = 2595 * np.log10(1 + (end_freq / 700))

        mel_points = np.linspace(start_mel, end_mel, filter_count+2)

        hz_points = (700 * (10 ** (mel_points / 2595) - 1))

        bin_s = np.floor((self.ft_n+1) * hz_points/self.rate)

        filter_bank = np.zeros((filter_count, self.ft_n // 2 + 1))
        for j in range(0, filter_count):
            for i in range(int(bin_s[j]), int(bin_s[j + 1])):
                filter_bank[j, i] = (i - bin_s[j]) / (bin_s[j + 1] - bin_s[j])
            for i in range(int(bin_s[j + 1]), int(bin_s[j + 2])):
                filter_bank[j, i] = (bin_s[j + 2] - i) / (bin_s[j + 2] - bin_s[j + 1])

        if plot:
            delta_frq = self.rate / self.ft_n
            fq_fft = np.arange(0, (self.ft_n//2)+1) * delta_frq

            plt.figure(figsize=(20, 10))
            for i in filter_bank:
                plt.plot(fq_fft, i)
            plt.title('Mel Filter Bank')
            plt.xlabel('Frequency (Hz)')
            plt.show()

        return filter_bank

    def log_frame_filter_bank(self, filter_bank):
        return np.log(np.dot(filter_bank, self.ft_sig**2) + 1e-12).T

    @classmethod
    def frames_dct(cls, filter_bank, c_count: int = 13):
        return dct(filter_bank, type=2, axis=1, norm='ortho')[:, :c_count]

    @classmethod
    def derivative(cls, v):
        diff = []
        for i in range(v.shape[0]):
            diff.append((v[min(i + 1, v.shape[0] - 1)] - v[max(i - 1, 0)]) +
                        2 * (v[min(i + 2, v.shape[0] - 2)] - v[max(i - 2, 0)]))

        diff = np.array(diff)
        diff /= 10
        return diff

    @classmethod
    def mfcc_derivatives(cls, MFCC):
        first_der = cls.derivative(MFCC)
        second_der = cls.derivative(first_der)

        return first_der, second_der

    def mfcc(self, cmvn: bool = True, derivatives: bool = True):
        self.pre_processing()
        self.frames_fft()

        filter_bank = self.filter_bank(plot=False)
        log_filter_bank = self.log_frame_filter_bank(filter_bank)
        MFCC = self.frames_dct(log_filter_bank)
        MFCC[:, 0] = self.short_time_energy(plot=False)

        if cmvn:
            voiced_frames = self.vad()
            MFCC = MFCC[voiced_frames]
            c_mean = np.mean(MFCC, axis=0)
            c_std = np.std(MFCC, axis=0)

            MFCC = (MFCC - c_mean) / c_std

        # np.save("Outputs/npy_files/mean_mfcc.npy", np.mean(MFCC, axis=0))
        # print(np.mean(MFCC, axis=0))
        # np.save("Outputs/npy_files/std_mfcc.npy", np.std(MFCC, axis=0))

        if derivatives:
            first_der, second_der = self.mfcc_derivatives(MFCC)
            MFCC = np.concatenate((MFCC, first_der, second_der), axis=1)

        # np.save("Outputs/npy_files/MFCC.npy", MFCC)
        return MFCC

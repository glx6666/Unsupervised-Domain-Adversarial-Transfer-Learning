import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

def ADSMPDM(IAS_i, D, K, f_fault, N, R, eta, fault_type):
    assert 1 <= D <= 10, 'D must be between 1 and 10'
    assert fault_type in ['bearing', 'gear'], 'fault_type must be "bearing" or "gear"'

    cutoff_freq = 0.8 * (1 / D)
    b, a = signal.butter(4, cutoff_freq, btype='low')
    IAS_i_filtered = signal.filtfilt(b, a, IAS_i)

    L = len(IAS_i_filtered)
    M = L // D
    IAS_i_truncated = IAS_i_filtered[:M * D]
    IAS_gD = np.mean(IAS_i_truncated.reshape(D, M), axis=0)

    if fault_type == 'bearing':
        N_s_m = (N * R) / (49 * D * f_fault) + 2 * D + 2
    else:
        N_s_m = 2 * D + 2
    N_s_m = int(np.floor(N_s_m))

    P = int(np.floor(N / (D * f_fault)))
    M_total = len(IAS_gD) - K * P - 2 * N_s_m
    DS_MPDM = np.zeros(M_total)

    delta_phi = 2 * np.pi / N
    denominator = D * delta_phi

    for m in range(M_total):
        sum_total = 0
        valid_k_count = 0

        for k in range(1, K + 1):
            q = m + (k - 1) * P
            h_start = max(0, int(q - eta * N_s_m))
            h_end = min(len(IAS_gD) - 1, int(q + eta * N_s_m))

            if h_start >= h_end:
                continue

            diffs = IAS_gD[h_start + 1:h_end + 1] - IAS_gD[h_start:h_end]
            sum_k = np.sum(diffs) / denominator
            num_points = len(diffs)

            if num_points > 0:
                sum_total += sum_k / num_points
                valid_k_count += 1

        if valid_k_count > 0:
            DS_MPDM[m] = sum_total / valid_k_count

    return DS_MPDM

# === 主程序 ===
dataFilePath = r'F:\数据\行星轮减速器故障实验数据\0\60\0V\test_N_0_0_60_2.mat'
data = sio.loadmat(dataFilePath)
var_name = list(data.keys())[-1]
signal_original = data[var_name].squeeze()[800:]

M = 2500
q = 1
N = 2500
f_reb = 2.7
fs = M * q
F = 40000000
delta_phi = 2 * np.pi / (M * q)
L = len(signal_original)

delta_x = signal_original / F
IAS_signal = delta_phi / delta_x
IAS_signal = IAS_signal - np.mean(IAS_signal)

# ADSMPDM 参数
D = 2
K = 8
eta = 2
fault_type = 'gear'
R = 5

# 运行算法
DS_MPDM = ADSMPDM(IAS_signal, D, K, f_reb, N, R, eta, fault_type)

# 阶次谱分析
order_resolution = 1 / (len(DS_MPDM) / (fs / D))
orders = np.arange(len(DS_MPDM)) * order_resolution
order_spectrum = np.abs(np.fft.fft(DS_MPDM))
order_spectrum = order_spectrum / np.max(order_spectrum)



# 创建一个 1 行 2 列的子图
plt.figure(figsize=(12, 8))

# 第一个子图：原始 IAS 信号
plt.subplot(3, 1, 1)
plt.plot(IAS_signal[800:20000], label='原始 IAS 信号', color='b')
plt.title('原始 IAS 信号')
plt.xlabel('样本点')
plt.ylabel('幅值')
plt.grid(True)

# 第二个子图：ADSMPDM 处理后信号
plt.subplot(3, 1, 2)
plt.plot(DS_MPDM[800:20000], label='ADSMPDM 处理信号', color='r', linewidth=2)
plt.title('ADSMPDM 处理后信号')
plt.xlabel('样本点')
plt.ylabel('幅值')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(orders, order_spectrum)
plt.xlim([0, 14])
plt.xlabel('Order [x]')
plt.ylabel('Amplitude')
plt.title('ADSMPDM 处理后阶次谱')
plt.grid(True)
# 调整布局
plt.tight_layout()
plt.show()
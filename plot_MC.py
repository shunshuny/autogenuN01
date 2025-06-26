import numpy as np
import matplotlib.pylab as plt
import os

# --- 設定 (元のコードと同じ) ---
logname = 'tank_two_MC'
xlog = logname + '_x.log'
tlog = logname + '_t.log'
logdir = os.path.join('.', 'logs', logname)
x_file = os.path.join(logdir, xlog)
t_file = os.path.join(logdir, tlog)

# --- データを読む (元のコードと同じ) ---
try:
    x_data = np.genfromtxt(x_file)
    t_data = np.genfromtxt(t_file)
except IOError:
    print(f"エラー: ログファイルが見つかりません。'{logdir}'のパスを確認してください。")
    exit()

# --- NaNを0に置換 (元のコードと同じ) ---
x_data[np.isnan(x_data)] = 0
t_data[np.isnan(t_data)] = 0

# --- シミュレーションパラメータ (元のコードと同じ) ---
num_simulations = 5
simulation_length = 100
sampling_time = 0.01
num_steps = int(simulation_length / sampling_time)

# 1. データの前処理
# 時間軸データは1シミュレーション分あれば良いため、先頭からnum_steps個だけ取り出す
t_axis = t_data[:num_steps]

# xのデータをシミュレーションごとに分割する
try:
    reshaped_x_data = x_data.reshape(num_simulations, num_steps, 2)
except ValueError:
    print(f"エラー: ログファイルのデータ数({x_data.shape[0]}行)が、")
    print(f"パラメータから期待されるデータ数({num_simulations * num_steps}行)と一致しません。")
    exit()


# 2. グラフを描画
# グラフの準備
plt.figure(figsize=(10, 6)) # 見やすいように少し図を大きくします

# forループで各シミュレーションの結果を重ねてプロット
for i in range(num_simulations):
    # 凡例にラベルが重複して表示されないよう、最初のループ(i=0)の時だけlabelを指定
    if i == 0:
        plt.plot(t_axis, reshaped_x_data[i, :, 0], color='blue', alpha=0.5, label='water level 1')
        plt.plot(t_axis, reshaped_x_data[i, :, 1], color='red', alpha=0.5, label='water level 2')
    else:
        # 2回目以降はラベルなしでプロット
        plt.plot(t_axis, reshaped_x_data[i, :, 0], color='blue', alpha=0.5)
        plt.plot(t_axis, reshaped_x_data[i, :, 1], color='red', alpha=0.5)

# (オプション) 平均値の軌道を太線でプロット
mean_x = np.mean(reshaped_x_data, axis=0)
plt.plot(t_axis, mean_x[:, 0], color='darkblue', linewidth=2, label='Mean water level 1')
plt.plot(t_axis, mean_x[:, 1], color='darkred', linewidth=2, label='Mean water level 2')



# --- グラフの体裁 (元のコードと同じ) ---
plt.xlabel('t', fontsize=14)
plt.ylabel('water level', fontsize=14)
plt.title('Monte Carlo Simulation Results', fontsize=16) # タイトルを追加
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig('monte_carlo_plot.png', dpi=300) # ファイル名と解像度を調整
plt.show()
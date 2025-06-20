import cgmres.tank
import cgmres.common
import numpy as np
from autogenu import RK4, Logger, Plotter
import sympy
from sympy import sin, cos, tan, exp, log, sinh, cosh, tanh, diff, sqrt, Matrix, integrate
import csv
import time

start = time.time() #コードの開始時刻

ocp_nominal = cgmres.tank.OCP()
# ocpL = cgmres.pareto.OCP()
# ocpS = cgmres.p3_minmax.OCP()

settings = cgmres.common.SolverSettings()
settings.sampling_time = 0.01
settings.zeta = 100
settings.finite_difference_epsilon = 1e-08
settings.max_iter = 0
settings.opterr_tol = 1e-06
settings.verbose_level = 1 # print opt error

horizon = cgmres.common.Horizon(Tf=1.0, alpha=0.5)


# 重み係数の変更など
# R_u = 20; R_u2 = 40; R_u3 = 80
# Q_p_val = 1; Q_p_val2 = 1; Q_p_val3 = 1
# ocpM.Q_s = ocpL.Q_s = ocpS.Q_s = 0.5; ocpM.Q_s2 = ocpL.Q_s2 = ocpS.Q_s2 = 0.5
# ocpM.Q_s3 = ocpL.Q_s3 = ocpS.Q_s3 = 0.5
# ocpM.ALPHA = ocpL.ALPHA = ocpS.ALPHA = 1; ocpM.qdash = ocpL.qdash = ocpS.qdash= 1
# ocpM.sigma = ocpL.sigma = ocpS.sigma= 5

player = 3

# 初期状態の設定
t0 = 0.0
x_o = np.array([5.0, 3.0, 4.0])
x0_nominal = np.concatenate([x_o, x_o, x_o])

# Initialize solution using zero horizon OCP solution (by racer_M)
initializer_nominal = cgmres.tank.ZeroHorizonOCPSolver(ocp_nominal, settings)
uc0_nominal = np.array([1.0, 0.5, 0.5, 0.5])
initializer_nominal.set_uc(uc0_nominal)
initializer_nominal.solve(t0, x0_nominal)

# Create MPC solver and set the initial solution (by racer_M)
mpc_nominal = cgmres.tank.MultipleShootingCGMRESSolver(ocp_nominal, horizon,settings)
mpc_nominal.set_uc(initializer_nominal.ucopt)
mpc_nominal.init_x_lmd(t0, x0_nominal)
mpc_nominal.init_dummy_mu()


# logファイル、ディレクトリ生成
logger = Logger(log_dir='logs/tank01', log_name='tank01')

# simple simulation with forward Euler ?? RK4 :Computes the next state by the 4th-order Runge-Kutta method. 
tsim = 500
sampling_time = settings.sampling_time
t = t0
x = x0_nominal.copy()

# ホライゾン上のｘの保存
# nominal_horizon = []

num_steps = int(tsim / sampling_time)

for i in range(num_steps):
    u_nominal = mpc_nominal.uopt[0]
    # buffer = []
    # for j in range(len(mpcM.xopt)):
    #     position = np.concatenate([mpcM.xopt[j][:3], mpcM.xopt[j][15:18], mpcM.xopt[j][30:33]])
    #     rounded = [round(k,3) for k in position] #小数点以下三桁に丸め
    #     buffer.extend(rounded) #ホライゾン上のｘを一つのリストに結合　サイズ：9*Nグリッド
    # M_horizon.append(buffer)
    # buffer = []
    # for j in range(len(mpcL.xopt)):
    #     position = np.concatenate([mpcL.xopt[j][:3], mpcL.xopt[j][15:18], mpcL.xopt[j][30:33]])
    #     rounded = [round(k,3) for k in position] #小数点以下三桁に丸め
    #     buffer.extend(rounded) #ホライゾン上のｘを一つのリストに結合　サイズ：9*Nグリッド
    # L_horizon.append(buffer)
    # buffer = []
    # for j in range(len(mpcS.xopt)):
    #     position = np.concatenate([mpcS.xopt[j][:3], mpcS.xopt[j][15:18], mpcS.xopt[j][30:33]])
    #     rounded = [round(k,3) for k in position] #小数点以下三桁に丸め
    #     buffer.extend(rounded) #ホライゾン上のｘを一つのリストに結合　サイズ：9*Nグリッド
    # S_horizon.append(buffer)

    # x_o, x_other =np.split(x, [3]) 
    x_nominal = RK4(ocp_nominal, t, sampling_time, x, u_nominal)
    # xL = RK4(ocpL, t, sampling_time, x_o, uL)
    # xS = RK4(ocpS, t, sampling_time, x_o, uS)
    
    # xM_p1, xM_p2, xM_p3, xM_other = np.split(xM,[15,30,45])
    # xL_p1, xL_p2, xL_p3 = np.split(xL,[15,30])
    # xS_p1, xS_p2, xS_p3 = np.split(xS,[15,30])
    # x_o = np.r_[xS_p1, xM_p2, xL_p3]
    # x = np.concatenate([x_o, x_o, x_o])
    x = x_nominal.copy()
    
    # uM_p1, uM_p2, uM_p3 = np.split(uM,[4,8])
    # uL_p1, uL_p2, uL_p3 = np.split(uL,[4,8])
    # uS_p1, uS_p2, uS_p3 = np.split(uS,[4,8])
    u = u_nominal.copy()
    
    
    opt_error = np.r_[mpc_nominal.opt_error()]
    
    logger.save(t, x, u, opt_error)
    
    t = t0 + (i + 1) * sampling_time

    mpc_nominal.update(t, x)
    print('t: ', t, ', x: ', x_o)
    # mpcL.update(t, x_o)
    # print('t: ', t, ', x: ', x_o)
    # mpcS.update(t, x_o)
    # print('t: ', t, ', x: ', x_o)
    # #print('t: ', t)
    
logger.close
# with open('logs/drone_p3MNP/drone_p3MNP_horizon_M.csv', 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=' ')
    
#     for row in M_horizon:
#         writer.writerow(row)
# with open('logs/drone_p3MNP/drone_p3MNP_horizon_L.csv', 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=' ')
#     for row in L_horizon:
#         writer.writerow(row)
# with open('logs/drone_p3MNP/drone_p3MNP_horizon_S.csv', 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=' ')
#     for row in S_horizon:
#         writer.writerow(row)



print("\n======================= MPC used in this simulation: =======================")
print(mpc_nominal)

end = time.time() #コードの終了時刻
time_diff = end - start
minute, second = time_diff // 60, time_diff % 60 #計算時間を時間と秒に分ける
print('All Calculate Time = ', minute, 'minute', second,'second') 

# time.sleep(5)

# plotter = Plotter(log_dir='logs/tank01', log_name='tank01', player = player)
# plotter.show()
# plotter.save()


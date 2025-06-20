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

horizon = cgmres.common.Horizon(Tf=1.0, alpha=0.0)


# 重み係数の変更など
ocp_nominal.Q1 = 1.0; ocp_nominal.Q2 = 100; ocp_nominal.Q3 = 10
ocp_nominal.r1 = 0.2; ocp_nominal.r2 = 0.2; ocp_nominal.r3 = 0.2; ocp_nominal.r0 = 1; ocp_nominal.r4 = 100
ocp_nominal.sf1 = 5; ocp_nominal.sf2 = 5; ocp_nominal.sf3 = 5

ocp_nominal.fb_eps = np.array([0.01])

player = 3

# 初期状態の設定
t0 = 0.0
x_o = np.array([5.0, 3.0, 4.0])
x0_nominal = np.concatenate([x_o, x_o, x_o])

# Initialize solution using zero horizon OCP solution
initializer_nominal = cgmres.tank.ZeroHorizonOCPSolver(ocp_nominal, settings)
uc0_nominal = np.array([1.0, 0.5, 0.5, 0.5, 0.01])
initializer_nominal.set_uc(uc0_nominal)
initializer_nominal.solve(t0, x0_nominal)

# Create MPC solver and set the initial solution
# mpc_nominal = cgmres.tank.MultipleShootingCGMRESSolver(ocp_nominal, horizon,settings)
mpc_nominal = cgmres.tank.SingleShootingCGMRESSolver(ocp_nominal, horizon,settings)
mpc_nominal.set_uc(initializer_nominal.ucopt)
# mpc_nominal.init_x_lmd(t0, x0_nominal)
# mpc_nominal.init_dummy_mu()


# logファイル、ディレクトリ生成
logger = Logger(log_dir='logs/tank01', log_name='tank01')

# simple simulation with forward Euler ?? RK4 :Computes the next state by the 4th-order Runge-Kutta method. 
tsim = 1000
sampling_time = settings.sampling_time
t = t0
x = x0_nominal.copy()

# ホライゾン上のｘの保存
# nominal_horizon = []

num_steps = int(tsim / sampling_time)

for i in range(num_steps):
    u_nominal = mpc_nominal.uopt[0] # 系列の最初を入力とする
    x_nominal = RK4(ocp_nominal, t, sampling_time, x, u_nominal) # xを次の状態へ更新
    mpc_nominal.update(t, x_nominal)

    logger.save(t, x, u_nominal, mpc_nominal.opt_error())

    t = t + sampling_time
    x = x_nominal
    x_o = x[:3]  # 3つの水位を抽出

    print('t: ', t, ', x: ', x_o)
    # mpcL.update(t, x_o)
    # print('t: ', t, ', x: ', x_o)
    # mpcS.update(t, x_o)
    # print('t: ', t, ', x: ', x_o)
    # #print('t: ', t)
    
logger.close

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


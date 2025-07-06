import cgmres.linear_PC
import cgmres.common
import numpy as np
from autogenu import RK4, Logger, Plotter, forward_euler
import sympy
from sympy import sin, cos, tan, exp, log, sinh, cosh, tanh, diff, sqrt, Matrix, integrate
import csv
import time

start = time.time() #コードの開始時刻

ocp_nominal = cgmres.linear_PC.OCP()

settings = cgmres.common.SolverSettings()
settings.sampling_time = 0.05
settings.zeta = 20
settings.finite_difference_epsilon = 1e-08
settings.max_iter = 0
settings.opterr_tol = 1e-06
settings.verbose_level = 1 # print opt error

horizon = cgmres.common.Horizon(Tf=1.0, alpha=0.5)


# 重み係数の変更など
# ocp_nominal.Q1 = 0.0; ocp_nominal.Q2 = 10; ocp_nominal.Q3 = 10
# ocp_nominal.r1 = 0.2; ocp_nominal.r2 = 0.2; ocp_nominal.r3 = 0.2; ocp_nominal.r0 = 0.01; ocp_nominal.r4 = 10
# ocp_nominal.sf1 = 0; ocp_nominal.sf2 = 10; ocp_nominal.sf3 = 10

# ocp_nominal.fb_eps = np.array([0.01])

player = 2
p_terms = 4
nu = player * p_terms  # 入力の数
nx = 2*p_terms

# 初期状態の設定
t0 = 0.0
x_o = np.array([-5,0,0,0, 6,0,0,0])
x0_nominal = np.concatenate([x_o, x_o])

# Initialize solution using zero horizon OCP solution
initializer_nominal = cgmres.linear_PC.ZeroHorizonOCPSolver(ocp_nominal, settings)
uc0_nominal = np.array([0.0]*nu)
initializer_nominal.set_uc(uc0_nominal)
initializer_nominal.solve(t0, x0_nominal)

# Create MPC solver and set the initial solution
# mpc_nominal = cgmres.linear_PC.MultipleShootingCGMRESSolver(ocp_nominal, horizon,settings)
mpc_nominal = cgmres.linear_PC.SingleShootingCGMRESSolver(ocp_nominal, horizon,settings)
mpc_nominal.set_uc(initializer_nominal.ucopt)
# mpc_nominal.init_x_lmd(t0, x0_nominal)
mpc_nominal.init_dummy_mu()


# logファイル、ディレクトリ生成
logger = Logger(log_dir='logs/linear_PC01', log_name='linear_PC01')

# simple simulation with forward Euler ?? RK4 :Computes the next state by the 4th-order Runge-Kutta method. 
tsim = 2
sampling_time = settings.sampling_time
t = t0
x = x0_nominal.copy()

# ホライゾン上のｘの保存
# nominal_horizon = []

num_steps = int(tsim / sampling_time)
keep_indices = [0,p_terms]

for i in range(num_steps):
    mpc_nominal.update(t, x)
    u_PC = mpc_nominal.uopt[0] # 系列の最初を入力とする
    u_nominal = np.zeros_like(u_PC)
    u_nominal[keep_indices] = u_PC[keep_indices]  # ノミナル入力のみを入れる．
    x_o = x[:nx]
    logger.save(t, x_o, u_nominal, mpc_nominal.opt_error())
    
    x_nominal = forward_euler(ocp_nominal, t, sampling_time, x, u_nominal) # xを次の状態へ更新
    # mpc_nominal.update(t, x_nominal)
    
    x = x_nominal
    t = t + sampling_time

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

# plotter = Plotter(log_dir='logs/linear_PC01', log_name='linear_PC01', player = player)
# plotter.show()
# plotter.save()


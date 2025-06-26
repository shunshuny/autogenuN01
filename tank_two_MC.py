import cgmres.tank_two
import cgmres.common
import numpy as np
import os
from autogenu import RK4, Logger, Plotter, forward_euler
import sympy
from sympy import sin, cos, tan, exp, log, sinh, cosh, tanh, diff, sqrt, Matrix, integrate
import csv
import time

def forward_euler_MC(t, dt, x: np.ndarray, u: np.ndarray):
    dx = np.zeros(np.size(x))
    dx[0] = (u[0] - u[1]*tanh(a*(x[0]-x[1]))*sqrt(2*g*(x[0]-x[1])*tanh(b*(x[0]-x[1]))) -q1*(1 + alpha*theta1))/S1
    dx[1] = (- u[1]*tanh(a*(x[1]-x[0]))*sqrt(2*g*(x[1]-x[0])*tanh(b*(x[1]-x[0]))) -q2*(1 + alpha*theta2))/S2
    dx[2] = dx[0]
    dx[3] = dx[1]
    return x + dx * dt

start = time.time() #コードの開始時刻

# ocp_nominal = cgmres.tank_two.OCP()

# settings = cgmres.common.SolverSettings()
# settings.sampling_time = 0.01
# settings.zeta = 100
# settings.finite_difference_epsilon = 1e-08
# settings.max_iter = 0
# settings.opterr_tol = 1e-06
# settings.verbose_level = 0 # print opt error

# horizon = cgmres.common.Horizon(Tf=1.5, alpha=0.5)


# # 重み係数の変更など
# ocp_nominal.Q1 = 0
# ocp_nominal.Q2 = 10
# ocp_nominal.r4 = 1
# ocp_nominal.sf1 = 5
# ocp_nominal.sf2 = 5
# ocp_nominal.x1_min = 4.5
# ocp_nominal.x2_min = 0.0

# # モデルのパラメータの設定
# a = 0.5
# b = 0.2
# g = 9.78
# S1 = 25.0
# S2 = 25.0
# q1 = 1.5
# q2 = 0.5

# player = 2

# # 初期状態の設定
# t0 = 0.0
# x_o = np.array([5.0, 3.0])
# x0_nominal = np.concatenate([x_o, x_o])

# # Initialize solution using zero horizon OCP solution
# initializer_nominal = cgmres.tank_two.ZeroHorizonOCPSolver(ocp_nominal, settings)
# uc0_nominal = np.array([1.0, 0.5, 0.01, 0.01])  # Initial guess for the control inputs
# initializer_nominal.set_uc(uc0_nominal)
# initializer_nominal.solve(t0, x0_nominal)


# # Create MPC solver and set the initial solution
# # mpc_nominal = cgmres.tank_two.MultipleShootingCGMRESSolver(ocp_nominal, horizon,settings)
# mpc_nominal = cgmres.tank_two.SingleShootingCGMRESSolver(ocp_nominal, horizon, settings)
# mpc_nominal.set_uc(initializer_nominal.ucopt)
# # mpc_nominal.init_x_lmd(t0, x0_nominal)
# mpc_nominal.init_dummy_mu()


# logファイル、ディレクトリ生成
log_dir='logs/tank_two_MC'
log_name='tank_two_MC'
logger = Logger(log_dir=log_dir, log_name=log_name)
theta_log = open(os.path.join(log_dir, log_name+"_theta.log"), mode='w')

# simple simulation with forward Euler ?? RK4 :Computes the next state by the 4th-order Runge-Kutta method. 
# tsim = 100
# sampling_time = settings.sampling_time
# t = t0
# x = x0_nominal.copy()

# num_steps = int(tsim / sampling_time)
num_simulations = 5 # モンテカルロシミュレーションの回数
all_thetas = []

for j in range(num_simulations):
    # 初期設定
    ocp_nominal = cgmres.tank_two.OCP()

    settings = cgmres.common.SolverSettings()
    settings.sampling_time = 0.01
    settings.zeta = 100
    settings.finite_difference_epsilon = 1e-08
    settings.max_iter = 0
    settings.opterr_tol = 1e-06
    settings.verbose_level = 0 # print opt error

    horizon = cgmres.common.Horizon(Tf=1.5, alpha=0.5)


    # 重み係数の変更など
    ocp_nominal.Q1 = 0
    ocp_nominal.Q2 = 10
    ocp_nominal.r4 = 1
    ocp_nominal.sf1 = 5
    ocp_nominal.sf2 = 5
    ocp_nominal.x1_min = 4.5
    ocp_nominal.x2_min = 0.0

    # モデルのパラメータの設定
    a = 0.5
    b = 0.2
    g = 9.78
    S1 = 25.0
    S2 = 25.0
    q1 = 1.5
    q2 = 0.5

    player = 2

    # 初期状態の設定
    t0 = 0.0
    x_o = np.array([5.0, 3.0])
    x0_nominal = np.concatenate([x_o, x_o])

    # Initialize solution using zero horizon OCP solution
    initializer_nominal = cgmres.tank_two.ZeroHorizonOCPSolver(ocp_nominal, settings)
    uc0_nominal = np.array([1.0, 0.5, 0.01, 0.01])  # Initial guess for the control inputs
    initializer_nominal.set_uc(uc0_nominal)
    initializer_nominal.solve(t0, x0_nominal)


    # Create MPC solver and set the initial solution
    # mpc_nominal = cgmres.tank_two.MultipleShootingCGMRESSolver(ocp_nominal, horizon,settings)
    mpc_nominal = cgmres.tank_two.SingleShootingCGMRESSolver(ocp_nominal, horizon, settings)
    mpc_nominal.set_uc(initializer_nominal.ucopt)
    # mpc_nominal.init_x_lmd(t0, x0_nominal)
    mpc_nominal.init_dummy_mu()
    
    tsim = 100
    sampling_time = settings.sampling_time
    t = t0
    x = x0_nominal.copy()

    num_steps = int(tsim / sampling_time)


    # MCのための初期化
    alpha = 0.1
    theta1 = np.random.uniform(-1, 1)
    theta2 = np.random.uniform(-1, 1)
    all_thetas.append([theta1, theta2])
    for i in range(num_steps):
        u_nominal = mpc_nominal.uopt[0] # 系列の最初を入力とする
        x_nominal = forward_euler_MC(t, sampling_time, x, u_nominal) # xを次の状態へ更新
        # x_nominal = RK4(ocp_nominal, t, sampling_time, x, u_nominal)
        mpc_nominal.update(t, x_nominal)

        x = x_nominal
        x_o = x[:2]  # 2つの水位を抽出
        logger.save(t, x_o, u_nominal, mpc_nominal.opt_error())


        t = t + sampling_time

        print('num_sim:', j+1, ', t: ', round(t, 3), ', x: ', x_o)

logger.close
theta_log_path = os.path.join(log_dir, log_name + "_theta.log")
np.savetxt(theta_log_path, np.array(all_thetas), delimiter=',')

print("\n======================= MPC used in this simulation: =======================")
print(mpc_nominal)

end = time.time() #コードの終了時刻
time_diff = end - start
minute, second = time_diff // 60, time_diff % 60 # 計算時間を時間と秒に分ける
print('All Calculate Time = ', minute, 'minute', second,'second') 

# time.sleep(5)

# plotter = Plotter(log_dir='logs/tank01', log_name='tank01', player = player)
# plotter.show()
# plotter.save()


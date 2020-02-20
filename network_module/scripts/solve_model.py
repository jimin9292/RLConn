
# coding: utf-8

import time
import os

import numpy as np
import scipy.io as sio

from scipy import integrate, signal, sparse, linalg

import paths as path

# Model Implementation ############################################################################################################################################
###################################################################################################################################################################

""" Number of Neurons """
N = 3

""" Cell membrane conductance (pS) """
Gc = 0.1
Gc_model = 0.2

""" Cell Membrane Capacitance"""
C = 0.015
C_model = 0.01

""" Gap Junctions (Electrical, 279*279) """
os.chdir(path.connectome_data_dir)

ggap = 1.0
Gg_Static = np.load('artificial_Gg.npy')

""" Synaptic connections (Chemical, 279*279) """
gsyn = 1.0
Gs_Static = np.load('artificial_Gs.npy')

""" Leakage potential (mV) """
Ec = -35.0
Ec_model = -45.0

""" Directionality (279*1) """
E = np.load('artificial_E.npy')
E = -48.0 * E
EMat = np.tile(np.reshape(E, N), (N, 1))

os.chdir(path.default_dir)

""" Synaptic Activity Parameters """
ar = 1.0/1.5 # Synaptic activity's rise time
ad = 5.0/1.5 # Synaptic activity's decay time
B = 0.125 # Width of the sigmoid (mv^-1)

Gg_Dynamic = Gg_Static
Gs_Dynamic = Gs_Static

mask_Healthy = np.ones(N, dtype = 'bool')

Vth_m_1 = np.load('V_th_model_1.npy')
Vth_m_2 = np.load('V_th_model_2.npy')
Vth_m_3 = np.load('V_th_model_3.npy')

def EffVth(Gg, Gs):

    Gcmat = np.multiply(Gc, np.eye(N))
    EcVec = np.multiply(Ec, np.ones((N, 1)))

    M1 = -Gcmat
    b1 = np.multiply(Gc, EcVec)

    Ggap = np.multiply(ggap, Gg)
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, N, N).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    Gs_ij = np.multiply(gsyn, Gs)
    s_eq = round((ar/(ar + 2 * ad)), 4)
    sjmat = np.multiply(s_eq, np.ones((N, N)))
    S_eq = np.multiply(s_eq, np.ones((N, 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, N, N).toarray()

    b3 = np.dot(Gs_ij, np.multiply(s_eq, E))

    M = M1 + M2 + M3

    global LL, UU, bb

    (P, LL, UU) = linalg.lu(M)
    bbb = -b1 - b3
    bb = np.reshape(bbb, N)

def EffVth_rhs(Iext, InMask):

    InputMask = np.multiply(Iext, InMask)
    b = np.subtract(bb, InputMask)

    Vth = linalg.solve_triangular(UU, linalg.solve_triangular(LL, b, lower = True, check_finite=False), check_finite=False)

    return Vth

def modify_Connectome(ablation_Mask):

    global Gg_Dynamic, Gs_Dynamic

    apply_Col = np.tile(ablation_Mask, (N, 1))
    apply_Row = np.transpose(apply_Col)

    apply_Mat = np.multiply(apply_Col, apply_Row)

    Gg_Dynamic = np.multiply(Gg_Static, apply_Mat)
    Gs_Dynamic = np.multiply(Gs_Static, apply_Mat)

    EffVth(Gg_Dynamic, Gs_Dynamic)

""" RHS Variants"""

def jimin_RHS(t, y):

    Vvec, SVec = np.split(y, 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(Gc, (Vvec - Ec))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (N, 1))
    GapCon = np.multiply(Gg_Dynamic, np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), EMat)
    SynapCon = np.multiply(np.multiply(Gs_Dynamic, np.tile(SVec, (N, 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(ar, (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-B*(np.subtract(Vvec, Vth)))))

    SynDrop = np.multiply(ad, SVec)

    """ Input Mask """
    Input = np.multiply(Iext, InMask)

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/C
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def jimin_RHS_Model(t, y):

    Vvec, SVec = np.split(y, 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(Gc_model, (Vvec - Ec_model))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (N, 1))
    GapCon = np.multiply(Gg_Dynamic, np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), EMat)
    SynapCon = np.multiply(np.multiply(Gs_Dynamic, np.tile(SVec, (N, 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(ar, (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-B*(np.subtract(Vvec, Vth)))))

    SynDrop = np.multiply(ad, SVec)

    """ Input Mask """
    Input = np.multiply(Iext, InMask)

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/C_model
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def jimin_RHS_Odeint(y, t):

    Vvec, SVec = np.split(y, 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(Gc_model, (Vvec - Ec_model))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (N, 1))
    GapCon = np.multiply(Gg_Dynamic, np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), EMat)
    SynapCon = np.multiply(np.multiply(Gs_Dynamic, np.tile(SVec, (N, 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(ar, (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-B*(np.subtract(Vvec, Vth)))))

    SynDrop = np.multiply(ad, SVec)

    """ Input Mask """
    Input = np.multiply(Iext, InMask)

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/C_model
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def run_Network(t_Start, t_Final, t_Delta, input_Mask, ablation_Mask = mask_Healthy, atol = 1e-3, mode = 'standard'):

    t0 = t_Start
    tf = t_Final
    dt = t_Delta

    global nsteps, InMask

    nsteps = int(np.floor((tf - t0)/dt) + 1)
    InMask = input_Mask

    """ define the connectivity """

    modify_Connectome(ablation_Mask)

    """ Input signal magnitude """
    global Iext

    Iext = 100000

    """ Calculate V_threshold """
    global Vth

    if mode == 'standard':

        Vth = EffVth_rhs(Iext, InMask)

    elif mode == 'model':

        if InMask[1] == 0.03:

            Vth = Vth_m_1

        elif InMask[1] == 0.04:

            Vth = Vth_m_2

        elif InMask[1] == 0.05:

            Vth = Vth_m_3

    InitCond = 10**(-4)*np.random.normal(0, 0.94, 2*N)

    """ Configuring the ODE Solver """

    if mode == 'standard':

        r = integrate.ode(jimin_RHS).set_integrator('vode', atol = atol, min_step = dt*1e-6, method = 'bdf', with_jacobian = True)

    elif mode == 'model':

        r = integrate.ode(jimin_RHS_Model).set_integrator('vode', atol = atol, min_step = dt*1e-6, method = 'bdf', with_jacobian = True)

    r.set_initial_value(InitCond, t0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    Traj_0 = np.zeros((nsteps, N))
    Traj_1 = np.zeros((nsteps, N))

    t[0] = t0
    Traj_0[0, :] = InitCond[:N]
    Traj_1[0, :] = InitCond[N:]

    """ Integrate the ODE(s) across each delta_t timestep """
    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        Traj_0[k, :] = r.y[:N]
        Traj_1[k, :] = r.y[N:]

        k += 1

    result_dict = {
            "t": t,
            "steps": nsteps,
            "voltage_mat": Traj_0,
            "syn_activity_mat": Traj_1,
            "V_threshold": Vth,
            }

    return result_dict

def add_Noise(result_dict, mean, v_var, s_var):

    t_arr = result_dict['t']
    vol_mat = result_dict['voltage_mat']
    syn_mat = result_dict['syn_activity_mat']
    V_th = result_dict['V_threshold']

    vol_mat_reshaped = vol_mat.transpose()
    syn_mat_reshaped = syn_mat.transpose()

    row_num_v = len(vol_mat_reshaped)
    row_num_s = len(syn_mat_reshaped)
    col_num = len(t_arr)

    V_noise_mat = np.zeros((row_num_v, col_num))
    S_noise_mat = np.zeros((row_num_s, col_num))

    std_v = np.sqrt(v_var)
    std_s = np.sqrt(s_var)

    for k in xrange(len(t_arr)):

        V_noise_mat[:, k] = np.random.normal(mean, std_v, row_num_v)
        S_noise_mat[:, k] = np.random.normal(mean, std_s, row_num_s)

    vol_mat_noised = np.add(vol_mat_reshaped, V_noise_mat)
    syn_mat_noised = np.add(syn_mat_reshaped, S_noise_mat)

    combined_noised = np.vstack([vol_mat_noised, syn_mat_noised])

    return t_arr, combined_noised, V_th

# EFK Implementation ##############################################################################################################################################
###################################################################################################################################################################

def compute_Init(x_init_true, var_list):

    x_init_list = []

    for k in xrange(len(x_init_true[:3])):

        x_init_list.append(np.random.normal(x_init_true[k], var_list[k], 1)[0])

    x_init_list.append(x_init_true[3])
    x_init_list.append(x_init_true[4])
    x_init_list.append(x_init_true[5])

    x_init_arr = np.asarray(x_init_list)[:, np.newaxis]

    return x_init_arr

def EKF_Jf(x_arr):

    variables_num = len(x_arr)

    x1 = x_arr[0][0]
    x2 = x_arr[1][0]
    x3 = x_arr[2][0]
    x4 = x_arr[3][0]
    x5 = x_arr[4][0]
    x6 = x_arr[5][0]

    E1 = E[0][0]
    E2 = E[1][0]
    E3 = E[2][0]

    Vth1 = Vth[0]
    Vth2 = Vth[1]
    Vth3 = Vth[2]

    df1x = np.zeros(variables_num)
    df2x = np.zeros(variables_num)
    df3x = np.zeros(variables_num)
    df4x = np.zeros(variables_num)
    df5x = np.zeros(variables_num)
    df6x = np.zeros(variables_num)

    df1x[0] = -(2*x5 + 8*x6 + Gc_model + 13) / C_model
    df1x[1] = 8 / C_model
    df1x[2] = 5 / C_model
    df1x[3] = 0
    df1x[4] = -2*(x1 - E2) / C_model
    df1x[5] = -8*(x1 - E3) / C_model

    df2x[0] = 8 / C_model
    df2x[1] = -(7*x4 + 3*x6 + Gc_model + 10) / C_model
    df2x[2] = 2 / C_model
    df2x[3] = -7*(x2 - E1) / C_model
    df2x[4] = 0
    df2x[5] = -3*(x2 - E3) / C_model

    df3x[0] = 5 / C_model
    df3x[1] = 2 / C_model
    df3x[2] = -(7*x4 + 7*x5 + Gc_model + 7) / C_model
    df3x[3] = -7*(x3 - E1) / C_model
    df3x[4] = -7*(x3 - E2) / C_model
    df3x[5] = 0

    df4x[0] = (-ar*(x4-1)*B*np.exp((B * x1) + (B * Vth1))) / np.power((np.exp(B * x1) + np.exp(B * Vth1)), 2)
    df4x[1] = 0
    df4x[2] = 0
    df4x[3] = -(ar*np.exp(B * x1) + ad*(np.exp(B * Vth1) + np.exp(B * x1))) / (np.exp(B * Vth1) + np.exp(B * x1))
    df4x[4] = 0
    df4x[5] = 0

    df5x[0] = 0
    df5x[1] = (-ar*(x5-1)*B*np.exp((B * x2) + (B * Vth2))) / np.power((np.exp(B * x2) + np.exp(B * Vth2)), 2)
    df5x[2] = 0
    df5x[3] = 0
    df5x[4] = -(ar*np.exp(B * x2) + ad*(np.exp(B * Vth2) + np.exp(B * x2))) / (np.exp(B * Vth2) + np.exp(B * x2))
    df5x[5] = 0

    df6x[0] = 0
    df6x[1] = 0
    df6x[2] = (-ar*(x6-1)*B*np.exp((B * x3) + (B * Vth3))) / np.power((np.exp(B * x3) + np.exp(B * Vth3)), 2)
    df6x[3] = 0
    df6x[4] = 0
    df6x[5] = -(ar*np.exp(B * x3) + ad*(np.exp(B * Vth3) + np.exp(B * x3))) / (np.exp(B * Vth3) + np.exp(B * x3))

    Jf = np.vstack([df1x, df2x, df3x, df4x, df5x, df6x])

    return Jf

def EKF_Jh(xk):

    H = np.eye(len(xk))

    return H

def compute_H(xk):

    return xk

def propagate_ODE(t, dt, x_hat):

    t_interval = np.asarray([t, t + dt])

    sol_row = integrate.odeint(jimin_RHS_Odeint, x_hat, t_interval)[1, :]
    sol = sol_row[:, np.newaxis]

    return sol

def EKF_Forecast(xk, Pk, tk, dt):

    # Forecast xk

    xk_RHS = np.reshape(xk, len(xk))
    new_xk = propagate_ODE(tk, dt, xk_RHS)

    # Forecast Pk

    Jf = EKF_Jf(xk)
    dPdt = np.dot(Jf, Pk) + np.dot(Pk, Jf.transpose())
    new_Pk = Pk + (np.multiply(dPdt, dt))

    # Forecast tk

    new_tk = tk + dt

    return new_xk, new_Pk, new_tk

def compute_K(xk, Pk, R):

    Jh = EKF_Jh(xk)

    term_0 = np.dot(Pk, Jh.transpose())
    term_1 = np.add(np.dot(np.dot(Jh, Pk), Jh.transpose()), R)

    K = np.dot(term_0, np.linalg.inv(term_1))

    return K

def EKF_Update(xk, Pk, yk, R):

    Jh = EKF_Jh(xk)
    h = compute_H(xk)

    K = compute_K(xk, Pk, R)
    identity = np.eye(len(xk))

    xk_updated = xk + np.dot(K, np.subtract(yk, h))
    Pk_updated_0 = np.subtract(identity, np.dot(K, Jh))
    Pk_updated = np.dot(Pk_updated_0, Pk)

    return xk_updated, Pk_updated

def run_EFK(x0, P0, t0, dt, R, y):

    nsteps = len(y[0, :])

    x_forecast_list = []
    P_forecast_list = []
    x_update_list = []
    P_update_list = []
    t_list = []

    x_forecast_list.append(x0)
    P_forecast_list.append(P0)
    x_update_list.append(x0)
    P_update_list.append(P0)
    t_list.append(t0)

    xk_forecast, Pk_forecast, tk_forecast = EKF_Forecast(x0, P0, t0, dt)
    x_forecast_list.append(xk_forecast)
    P_forecast_list.append(Pk_forecast)
    t_list.append(tk_forecast)

    k = 1

    while k < nsteps:

        measured_y = y[:, k][:, np.newaxis]

        xk = x_forecast_list[-1]
        Pk = P_forecast_list[-1]
        tk = t_list[-1]

        # Update

        xk_updated, Pk_updated = EKF_Update(xk, Pk, measured_y, R)
        x_update_list.append(xk_updated)
        P_update_list.append(Pk_updated)

        # Propagate

        xk_next, Pk_next, tk_next = EKF_Forecast(xk_updated, Pk_updated, tk, dt)
        x_forecast_list.append(xk_next)
        P_forecast_list.append(Pk_next)
        t_list.append(tk_next)

        k += 1

    return x_forecast_list, P_forecast_list, x_update_list, P_update_list

def compute_MSE(v_est_mat, v_real_mat):

    nsteps = len(v_real_mat[0, :])
    mse = np.divide(np.power(np.subtract(v_est_mat, v_real_mat), 2).sum(axis = 1), float(nsteps))

    return mse












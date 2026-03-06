# reproduce Linshiz-Titi-2006, ref to helicityhu
# helicity, cross helicity, energy
# temporal convergence for 3d MHD
from firedrake import *
import csv
import os
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from mpi4py import MPI
from tabulate import tabulate

nu = Constant(1)
eta = Constant(1)
S = Constant(1)

# solver parameter
ICNTL_14 = 5000
tele_reduc_fac = int(MPI.COMM_WORLD.size/4)
if tele_reduc_fac < 1:
    tele_reduc_fac = 1

lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "telescope",
    "pc_telescope_reduction_factor": tele_reduc_fac,
    "pc_telescope_subcomm_type": "contiguous",
    "telescope_pc_type": "lu",
    "telescope_pc_factor_mat_solver_type": "mumps",
    "telescope_pc_factor_mat_mumps_icntl_14": ICNTL_14,
}
sp = lu

# spatial parameters
nref = 0

# Storage for errors
errors_u = []
errors_P = []
errors_B = []
rates_u = []
rates_P = []
rates_B = []
T = 1.0
L_values = [2, 4, 8]

for L in L_values:
    dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    mesh = UnitCubeMesh(L, L, L, distribution_parameters=dp)
    x, y ,z0= SpatialCoordinate(mesh)

    # spatial discretization
    Vg = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 1)
    Vd = FunctionSpace(mesh, "RT", 1)
    Vc = FunctionSpace(mesh, "N1curl", 1)
    Vn = FunctionSpace(mesh, "DG", 0)

    alpha = CellDiameter(mesh)

    # (u, P, u_b, w, B, E, j, H)
    Z = MixedFunctionSpace([Vc, Q, Vc, Vc, Vd, Vc, Vc, Vc])
    z = Function(Z)
    z_test = TestFunction(Z)
    z_prev = Function(Z)
    z_mean = Function(Z)

    (u, P, u_b, w, B, E, j, H) = split(z)
    (ut, Pt, u_bt, wt, Bt, Et, jt, Ht) = split(z_test)
    (up, Pp, u_bp, wp, Bp, Ep, jp, Hp) = split(z_prev)

    # solve for u_b_init
    def u_b_solver(u):
        u_init = Function(Vc).interpolate(u)
        u_b = Function(Vc)
        v = TestFunction(Vc)
        F = inner(u_b, v) * dx + alpha**2 * inner(curl(u_b), curl(v)) * dx - inner(u_init, v) * dx
        sp_ub = {  
               "ksp_type":"gmres",
               "pc_type": "ilu",
        }
        sp_riesz = {
             "mat_type": "nest",
            "snes_type": "ksponly",
            "snes_monitor": None,
            "ksp_monitor": None,
            "ksp_max_it": 1000,
            "ksp_norm_type": "preconditioned",
            "ksp_type": "minres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_atol": 1.0e-5,
            "ksp_rtol": 1.0e-5,
            "ksp_minres_nutol": 1E-8,
            "ksp_convergence_test": "skip",

        }
        
        def riesz_u_b(u, v):
            return inner(u, v) * dx + alpha **2 * inner(curl(u), curl(v)) * dx
        
        u_b0 = TrialFunction(Vc)
        u_b1 = TestFunction(Vc)
        Jp_riesz = riesz_u_b(u_b0, u_b1)

        bcs0 = [DirichletBC(Vc, 0, "on_boundary")]
        pb0 = NonlinearVariationalProblem(F, u_b, bcs0, Jp = Jp_riesz)
        solver0 = NonlinearVariationalSolver(pb0, solver_parameters = sp_riesz, options_prefix = "solve curlcurl for u_b") 
        solver0.solve()
        return u_b

    # time 
    t = Constant(0) 
    dt = Constant(0.1)

    def h(x):
        return (x**2 - x) **2

    def h_d(x):
        return 4 * x**3 - 6 * x **2 + 2 * x


    g_1 = 4 - 2* t 
    g_2 = 1 + t
    g_3 = 1 - t

    g_1_t = -2
    g_2_t = 1
    g_3_t = -1

    u_x = -g_1 * h_d(x) * h(y) * h(z0)
    u_y = -g_2 * h(x) * h_d(y) * h(z0)
    u_z = -g_3 * h(x) * h(y) * h_d(z0)

    u_x_t = -g_1_t * h_d(x) * h(y) * h(z0)
    u_y_t = -g_2_t * h(x) * h_d(y) * h(z0)
    u_z_t = -g_3_t * h(x) * h(y) * h_d(z0)

    u_ex = as_vector([u_x, u_y, u_z])
    u_exact_t = as_vector([u_x_t, u_y_t, u_z_t])
    B_exact_t = curl(u_exact_t)
    B_ex = curl(u_ex)
    w_ex = Function(Vc).interpolate(curl(u_ex))
    H_ex = Function(Vc).interpolate(B_ex)
    j_ex = curl(B_ex) 

    u_b_ex = u_b_solver(u_ex)
    E_ex = nu * j_ex - Function(Vc).interpolate(cross(u_b_ex, H_ex))
    P_ex =h(x) * h(y) * h(z0)

    # source term
    f1 = u_exact_t - cross(u_b_ex, w_ex) + nu * curl(curl(u_ex)) - S * cross(j_ex, H_ex) + grad(P_ex)
    f2 = B_exact_t + curl(E_ex)
    f3 = E_ex + cross(u_b_ex, H_ex) - eta * j_ex

    z_prev.sub(0).interpolate(u_ex)
    z_prev.sub(2).interpolate(u_b_ex)
    z_prev.sub(4).interpolate(B_ex) 
    z.assign(z_prev)

    u_avg = (u + up)/2
    B_avg = (B + Bp)/2
    u_b_avg = (u_b + u_bp)/2
    P_avg = P
    j_avg = j
    H_avg = H
    w_avg = w
    E_avg = E

    F= (    # u
          inner((u - up)/dt, ut) * dx
        - inner(cross(u_b_avg, w_avg), ut) * dx # advection term
        + inner(grad(P_avg), ut) * dx
        + nu * inner(curl(u_avg), curl(ut)) * dx
        - S * inner(cross(j_avg, H_avg), ut) * dx
        # p
        + inner(u_avg, grad(Pt)) * dx
        - inner(f1, ut) * dx
        # u_b
        + inner(u_b, u_bt) * dx
        + alpha**2 * inner(curl(u_b), curl(u_bt)) * dx
        - inner(u, u_bt) * dx
        # w
        + inner(w_avg, wt) * dx
        - inner(curl(u_avg), wt) * dx
        # B
        + inner((B - Bp)/dt, Bt) * dx
        + inner(curl(E_avg), Bt) * dx
        - inner(f2, Bt) * dx
        # E
        + inner(E_avg, Et) * dx
        + inner(cross(u_b_avg, H_avg), Et) * dx
        - eta * inner(j_avg, Et) * dx
        - inner(f3, Et) * dx
        # j 
        + inner(j_avg, jt) * dx
        - inner(B_avg, curl(jt)) * dx
        # H
        + inner(H_avg, Ht) * dx
        - inner(B_avg, Ht) * dx
    )

    dirichlet_ids = ("on_boundary",)
    #bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]
    # (u, P, u_b, w, B, E, j, H)

    bcs = [
        DirichletBC(Z.sub(0), u_ex, "on_boundary"),
        DirichletBC(Z.sub(1), P_ex, "on_boundary"),
        DirichletBC(Z.sub(2), u_b_ex, "on_boundary"),
        DirichletBC(Z.sub(3), w_ex, "on_boundary"),
        DirichletBC(Z.sub(4), B_ex, "on_boundary"),
        DirichletBC(Z.sub(5), E_ex, "on_boundary"),
        DirichletBC(Z.sub(6), j_ex, "on_boundary"), 
        DirichletBC(Z.sub(7), H_ex, "on_boundary"), 
    ]

    pb = NonlinearVariationalProblem(F, z, bcs)
    solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

    while (float(t) < float(T-dt)+1.0e-10):
        t.assign(t+dt)
        dofs = Z.dim()
        dofs_per_core = dofs / COMM_WORLD.size
        if mesh.comm.rank == 0:
            print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt)}, T = {T}, baseN = {L}, nref = {nref}, nu = {float(nu)}, dofs = {dofs}, dofs_per_core = {dofs_per_core}", flush=True)
        solver.solve()
     
        z_prev.assign(z)

    t.assign(T)
    z_mean.assign(0.5 * (z + z_prev)) 
    u_error = norm(z.sub(0) - u_ex, "L2")
    B_error = norm(z.sub(4) - B_ex, "L2")
    t.assign(T - dt/2)
    P_error = norm(z_mean.sub(1) - P_ex, "L2")
    print(f"error_u:{u_error}")
    print(f"error_P:{P_error}")
    print(f"error_B:{B_error}")
    errors_u.append(u_error)
    errors_P.append(P_error)
    errors_B.append(B_error)

L_values = [1/L for L in L_values]
# Compute convergence rates
for i in range(1, len(L_values)):
    rate_u = np.log(errors_u[i] / errors_u[i-1]) / np.log(L_values[i] / L_values[i-1])
    rate_P = np.log(errors_P[i] / errors_P[i-1]) / np.log(L_values[i] / L_values[i-1])
    rate_B = np.log(errors_B[i] / errors_B[i-1]) / np.log(L_values[i] / L_values[i-1])
    rates_u.append(rate_u)
    rates_P.append(rate_P)
    rates_B.append(rate_B)

# Print results
headers = ["h", "Error (u)", "Rate (u)", "Error (P)", "Rate(P)", "Error (B)", "Rate (B)"]
table_data = []
for i in range(len(L_values)):
    if i == 0:
        table_data.append([L_values[i], errors_u[i], "-", errors_P[i], "-", errors_B[i], "-"])
    else:
        table_data.append([L_values[i], errors_u[i], rates_u[i-1], errors_P[i], rates_P[i-1], errors_B[i], rates_B[i-1]])

print("\nSpatial Convergence Results:")
print(tabulate(table_data, headers=headers, floatfmt=".4e"))
print(tabulate(table_data, headers=headers, tablefmt="latex"))

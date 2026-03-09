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
def run_simulation(dt, L):
    dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    mesh = UnitCubeMesh(L, L, L, distribution_parameters=dp)
    x, y ,z0= SpatialCoordinate(mesh)

    # spatial discretization
    Vg = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 2)
    Vd = FunctionSpace(mesh, "RT", 2)
    Vc = FunctionSpace(mesh, "N1curl", 2)
    Vn = FunctionSpace(mesh, "DG", 1)

    #alpha = CellDiameter(mesh)
    alpha = Constant(0.1)
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
    t_half = Constant(0)
    dt = Constant(dt)

    s  = sin(pi*x) * sin(pi*y) * sin(pi*z0)
    sx = cos(pi*x) * sin(pi*y) * sin(pi*z0)
    sy = sin(pi*x) * cos(pi*y) * sin(pi*z0)
    sz = sin(pi*x) * sin(pi*y) * cos(pi*z0)

    phi = as_vector([sy - sz, sz - sx, sx - sy])

    lmbda = 1 + 3 * alpha**2 * pi**2
    mu    = Constant(5)  

    u_ex     = pi * exp(-nu * t) * phi
    u_ex_t   = -nu * pi * exp(-nu * t) * phi          # du/dt

    u_b_ex   = pi * exp(-nu * t) / lmbda * phi        # (I - alpha^2 Delta)^{-1} u
    u_b_ex_t = -nu * pi * exp(-nu * t) / lmbda * phi  # du_b/dt（仅供参考，不直接用）

    B_ex     = pi * exp(-mu * t) * phi
    B_ex_t   = -mu * pi * exp(-mu * t) * phi          # dB/dt

    H_ex     = B_ex
    j_ex     = curl(B_ex)                             # = curl(pi * exp(-mu*t) * phi)
    w_ex     = curl(u_ex)                             # = curl(pi * exp(-nu*t) * phi)

    E_ex     = (1/eta) * j_ex - cross(u_b_ex, H_ex)
    P_ex     = Constant(0)     
    # the above are evaluated at t since for initial BC
    # source term evaluate at t_half
    u_h    = pi * exp(-nu * t_half) * phi
    u_h_t    = -nu * pi * exp(-nu * t_half) * phi
    u_b_h  = pi * exp(-nu * t_half) / lmbda * phi
    B_h    = pi * exp(-mu * t_half) * phi
    B_h_t    = -mu * pi * exp(-mu * t_half) * phi
    w_h    = curl(u_h)
    j_h    = curl(B_h)
    E_h    = eta * j_h - cross(u_b_h, B_h)

    f1 = u_h_t - cross(u_b_h, w_h) + nu * curl(curl(u_h)) - S*cross(j_h, B_h)
    f2 = B_h_t + curl(E_h)
    f3 = E_h + cross(u_b_h, B_h) - eta * j_h
    
    z_prev.sub(0).interpolate(u_ex)
    z_prev.sub(1).interpolate(P_ex)
    z_prev.sub(2).interpolate(u_b_ex)
    z_prev.sub(3).interpolate(w_ex)
    z_prev.sub(4).interpolate(B_ex) 
    z_prev.sub(5).interpolate(E_ex) 
    z_prev.sub(6).interpolate(j_ex) 
    z_prev.sub(7).interpolate(H_ex) 
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

    #bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]
    pb = NonlinearVariationalProblem(F, z, bcs)
    solver = NonlinearVariationalSolver(pb, solver_parameters = sp)
    while (float(t) < float(T-dt)+1.0e-10):
        t_half.assign(float(t) + 0.5 * float(dt))
        t.assign(t+dt)
        dofs = Z.dim()
        dofs_per_core = dofs / COMM_WORLD.size
        if mesh.comm.rank == 0:
            print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt)}, T = {T}, baseN = {L}, nref = {nref}, nu = {float(nu)}, dofs = {dofs}, dofs_per_core = {dofs_per_core}", flush=True)
        solver.solve()
     
        z_prev.assign(z)

    t.assign(T)
    def remove_mean(p):
        mean = assemble(p * dx) / assemble(1 * dx(mesh))
        return p - Constant(mean)
    #t.assign(T - dt/2)
    #z_mean.assign(0.5 * (z + z_prev)) 
    u_error = norm(z.sub(0) - u_ex, "L2")
    B_error = norm(z.sub(4) - B_ex, "L2")
    P_error = norm(remove_mean(z.sub(1)) - P_ex, "L2")
    print(f"error_u:{u_error}")
    print(f"error_P:{P_error}")
    print(f"error_B:{B_error}")

    if PETSc.COMM_WORLD.rank == 0:
        print(f"h: {1/L}, u_error: {u_error}, p_error: {P_error}, B_error: {B_error}")

    if PETSc.COMM_WORLD.rank == 0:
        filename = "all_errors.csv"
        write_header = not os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            if write_header:
                f.write("dt,error_u,error_p,error_B\n")
            f.write(f"{float(dt)},{u_error},{P_error},{B_error}\n")

if PETSc.COMM_WORLD.rank == 0:
    if os.path.exists("all_errors.csv"):
        os.remove("all_errors.csv")
dt_values = [1/2, 1/4, 1/8]
Ls = [8, 8, 8]
for dt, L in zip(dt_values, Ls):
    run_simulation(dt, L)



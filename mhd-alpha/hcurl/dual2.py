# dual2 scheme
# u2 B2
# see whether it preserves the three structures
from firedrake import *
import csv
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import os
nu = Constant(0)
eta = Constant(0)
S = Constant(1)
def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]


def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])


def scurl(x):
    return x[1].dx(0) - x[0].dx(1)


def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])


def acurl(x):
    return as_vector([
                     x[2].dx(1),
                     -x[2].dx(0),
                     x[1].dx(0) - x[0].dx(1)
                     ])
def helicity_c(u, B):
    return assemble(inner(u, B)*dx)

def div_u(u):
    return norm(div(u), "L2")

def div_B(B):
    return norm(div(B), "L2")

def energy_uB(u, u_b, B):
    return 0.5 * assemble(inner(u, u_b) * dx + S * inner(B, B) * dx)

def compute_A(A):
    return assemble(inner(A, A) * dx)

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
baseN = 100
nref = 0
mesh = PeriodicUnitSquareMesh(baseN, baseN)
mesh.coordinates.dat.data[:] *= 2 * pi

x, y = SpatialCoordinate(mesh)

# spatial discretization
Vg = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
Vd = FunctionSpace(mesh, "RT", 1)
Vc = FunctionSpace(mesh, "N1curl", 1)
Vn = FunctionSpace(mesh, "DG", 0)

# time 
t = Constant(0) 
T = 20.0
dt = Constant(0.1)

alpha = CellDiameter(mesh)
    

# (u2, P3, w1, u_b2, w_b1, A1, B2, j1)
Z = MixedFunctionSpace([Vd, Vn, Q, Vd, Q, Q, Vd, Q])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, P, w, u_b, w_b, A, B,j) = split(z)
(ut, Pt, wt, u_bt, w_bt, At, Bt, jt) = split(z_test)
(up, Pp, wp, u_bp, w_bp, Ap, Bp, jp) = split(z_prev)

# Biskamp-Welter-1989
def v_grad(x):
    return as_vector([-x.dx(1), x.dx(0)])

phi = cos(x + 1.4) + cos(y + 0.5)
psi = cos(2 * x + 2.3) + cos(y + 4.1)

#phi = cos(x) + cos(y)
#psi = 0.5* cos(2 * x) + cos(y)

u_init = v_grad(psi)
B_init = v_grad(phi)
# v_grad = -vcurl
# A = -phi
A_init = -phi
alpha = CellDiameter(mesh)

# compute the value of meshsize alpha
def mesh_sizes(mh):
     mesh_size = []
     for msh in mh:
         DG0 = FunctionSpace(msh, "DG", 0)
         h = Function(DG0).interpolate(CellDiameter(msh))
         with h.dat.vec as hvec:
             _, maxh = hvec.max()
         mesh_size.append(maxh)
     return mesh_size

# solve for u_b_init
def u_b_solver(u):
    Z_b = MixedFunctionSpace([Vd, Q]) # u_b, w_b
    z_b = Function(Z_b)
    z_t = TestFunction(Z_b)

    (u_b, w_b) = split(z_b)
    (u_bt, w_bt) = split(z_t)
    F_b = (
        # u_b
          inner(u_b, u_bt) * dx
        + alpha **2 * inner(vcurl(w_b), u_bt) * dx
        - inner(u, u_bt) * dx
        # w_b
        + inner(w_b, w_bt) * dx
        - inner(u_b, vcurl(w_bt)) * dx
    )

    bcs0 = [
        DirichletBC(Z_b.sub(0), 0, "on_boundary"),
        DirichletBC(Z_b.sub(1), 0, "on_boundary")
    ]
    bcs0 = None
    pb0 = NonlinearVariationalProblem(F_b, z_b, bcs0)
    solver0 = NonlinearVariationalSolver(pb0, solver_parameters = lu, options_prefix = "solve curlcurl for u_b") 
    solver0.solve()
    return z_b.sub(0), z_b.sub(1)


def project_ic(B_init):
    # Need to project the initial conditions
    # such that div(B) = 0 and B·n = 0
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    (B, p) = split(zp)
    dirichlet_ids = ("on_boundary",)
    bcp = [DirichletBC(Zp.sub(0), 0, subdomain) for subdomain in dirichlet_ids]
    # Write Lagrangian
    L = (
          0.5*inner(B, B)*dx
        - inner(B_init, B)*dx
        - inner(p, div(B))*dx
        )

    Fp = derivative(L, zp, TestFunction(Zp))
    spp = {
        "mat_type": "nest",
        "snes_type": "ksponly",
        "snes_monitor": None,
        "ksp_monitor": None,
        "ksp_max_it": 1000,
        "ksp_norm_type": "preconditioned",
        "ksp_type": "minres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_pc_type": "cholesky",
        "fieldsplit_pc_factor_mat_solver_type": "mumps",
        "ksp_atol": 1.0e-5,
        "ksp_rtol": 1.0e-5,
        "ksp_minres_nutol": 1E-8,
        "ksp_convergence_test": "skip",
    }
    gamma = Constant(1E5)
    Up = 0.5*(inner(B, B) + inner(div(B) * gamma, div(B)) + inner(p * (1/gamma), p))*dx
    Jp = derivative(derivative(Up, zp), zp)
    solve(Fp == 0, zp, bcp, Jp=Jp, solver_parameters=spp,
          options_prefix="B_init_div_free_projection")
 
    return zp.subfunctions[0]

def helicity_m(B):
    return 0, 0

u_init_proj = project_ic(u_init)
u_b_init, w_b_init = u_b_solver(u_init_proj)
B_init_proj = project_ic(B_init)
#A_init = helicity_m(B_init_proj)[0]

z_prev.sub(0).interpolate(u_init_proj)
z_prev.sub(3).interpolate(u_b_init)
#z_prev.sub(4).interpolate(w_b_init)
z_prev.sub(5).interpolate(A_init)  
#z_prev.sub(6).interpolate(B_init_proj)  
z.assign(z_prev)

u_avg = (u + up)/2
A_avg = (A + Ap)/2
B_avg = B 
u_b_avg = (u_b + u_bp)/2 
P_avg = P
j_avg = j
w_avg = w
w_b_avg = w_b

F =( 
    # u
      inner((u - up)/dt, ut) * dx
    - inner(vcross(u_b_avg, w_avg), ut) * dx # advection term
    - inner(P_avg, div(ut)) * dx
    + nu * inner(vcurl(w_avg), ut) * dx
    + S * inner(vcross(B_avg, j_avg), ut) * dx
    # p
    - inner(div(u_avg), Pt) * dx
    # w  
    + inner(w_avg, wt) * dx
    - inner(u_avg, vcurl(wt)) * dx
    # u_b
    + inner(u_b, u_bt) * dx
    + alpha**2 * inner(vcurl(w_b), u_bt) * dx
    - inner(u, u_bt) * dx
    # w_b
    + inner(w_b_avg, w_bt) * dx
    - inner(u_b_avg, vcurl(w_bt)) * dx
    # A
    + inner((A - Ap)/dt, At) * dx
    + eta * inner(j_avg, At) * dx
    - inner(scross(u_b_avg, B_avg), At) * dx
    #B
    + inner(B_avg, Bt) * dx
    - inner(vcurl(A_avg), Bt) * dx
    #j 
    + inner(j_avg, jt) * dx
    - inner(B_avg, vcurl(jt)) * dx
)

dirichlet_ids = ("on_boundary",)
bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]
bcs = None

(u_, P_, w_, u_b_, w_b_, A_, B_, j_) = z.subfunctions
u_.rename("Velocity")
P_.rename("Pressure")
u_b_.rename("filteredVelocity")
w_.rename("Vorticity")
B_.rename("MagneticField")
A_.rename("MagneticPotential")
j_.rename("Current")

pvd = VTKFile("output/mhd-alpha.pvd")


def norm_inf(u):
    with u.dat.vec_ro as u_v:
        u_max = u_v.norm(PETSc.NormType.INFINITY)
    return u_max

def compute_ens(w, j):
    w_max=norm_inf(w)
    j_max=norm_inf(j)
    return w_max, j_max, float(w_max) + float(j_max) 

pb = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

timestep = 0
data_filename = "output/data.csv"
fieldnames = ["t", "divu", "divB", "energy", "helicity_c", "helicity_m", "ens_total", "w_max", "j_max", "A"]
# store the mesh info alpha and k_alpha
if mesh.comm.rank == 0:
    alpha_val = mesh_sizes(mesh)[0]
    k_alpha_val = 2 * pi / alpha_val

    with open("output/mesh_info.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "k_alpha"])
        writer.writerow([alpha_val, k_alpha_val])

    print(f"[mesh] alpha = {alpha_val}, k_alpha = {k_alpha_val}")
if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

#u_b, w_b = u_b_solver(z_prev.sub(0))
energy = energy_uB(z_prev.sub(0), z_prev.sub(3), z_prev.sub(6)) #u, u_b, B
crosshelicity = helicity_c(z_prev.sub(0), z_prev.sub(6)) # u, B
A_fn, maghelicity = helicity_m(z_prev.sub(6)) # B
mean_square_potential = compute_A(z_prev.sub(5))
divu = div_u(z_prev.sub(0))
divB = div_B(z_prev.sub(6))
# monitor
w_max, j_max, ens_total = compute_ens(z_prev.sub(2), z_prev.sub(7)) # w, j

if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "divu": float(divu),
        "divB": float(divB),
        "energy": float(energy),
        "helicity_c": float(crosshelicity),
        "helicity_m": float(maghelicity),
        "ens_total": float(ens_total),
        "w_max": float(w_max), 
        "j_max": float(j_max), 
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    dofs = Z.dim()
    dofs_per_core = dofs / COMM_WORLD.size
    if mesh.comm.rank == 0:
        print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt)}, T = {T}, baseN = {baseN}, nref = {nref}, nu = {float(nu)}, dofs = {dofs}, dofs_per_core = {dofs_per_core}", flush=True)
    solver.solve()
    #u_b, w_b= u_b_solver(z.sub(0)) 
    energy = energy_uB(z.sub(0), z.sub(3), z.sub(6)) #u, u_b, B
    crosshelicity = helicity_c(z.sub(0), z.sub(6)) # u, u_b, B
    A_fn, maghelicity = helicity_m(z.sub(6)) # B
    mean_square_potential = compute_A(z_prev.sub(5))

    divu = div_u(z.sub(0))
    divB = div_B(z.sub(6))
    # monitor
    w_max, j_max, ens_tol = compute_ens(z.sub(2), z.sub(7)) # w, j



    if mesh.comm.rank == 0:
        row = {
        "t": float(t),
        "divu": float(divu),
        "divB": float(divB),
        "energy": float(energy),
        "helicity_c": float(crosshelicity),
        "helicity_m": float(maghelicity),
        "ens_total": float(ens_total),
        "w_max": float(w_max), 
        "j_max": float(j_max), 
        "A": float(mean_square_potential),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
    if mesh.comm.rank == 0:
        print(row)
    
    if mesh.comm.rank == 0:
        #spectrum_and_save(z.sub(0), A_fn, z.sub(4), float(t))  # 会写 CSV 并存图
        pvd.write(*z.subfunctions, time=float(t))
    timestep += 1
    z_prev.assign(z)


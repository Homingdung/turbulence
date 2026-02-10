# reproduce Linshiz-Titi-2006, ref to helicityhu
# helicity, cross helicity, energy
from firedrake import *
import csv
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import os
nu = Constant(1e-3)
eta = Constant(1e-3)
S = Constant(1)

def helicity_c(u, B):
    return assemble(inner(u, B)*dx)

def div_u(u):
    return norm(div(u), "L2")

def div_B(B):
    return norm(div(B), "L2")

def energy_uB(u, u_b, B):
    return 0.5 * assemble(inner(u, u_b) * dx + S * inner(B, B) * dx)


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
baseN = 4
nref = 0
mesh = PeriodicUnitCubeMesh(baseN, baseN, baseN)
mesh.coordinates.dat.data[:] *= 2 * pi

x, y, z0 = SpatialCoordinate(mesh)

# spatial discretization
Vg = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
Vd = FunctionSpace(mesh, "RT", 1)
Vc = FunctionSpace(mesh, "N1curl", 1)
Vn = FunctionSpace(mesh, "DG", 0)

# time 
t = Constant(0) 
T = 1.0
dt = Constant(0.01)

alpha = CellDiameter(mesh)
    

# (u, P, u_b, w, B, E, j, H)
Z = MixedFunctionSpace([Vc, Q, Vc, Vc, Vd, Vc, Vc, Vc])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, P, u_b, w, B, E, j, H) = split(z)
(ut, Pt, u_bt, wt, Bt, Et, jt, Ht) = split(z_test)
(up, Pp, u_bp, wp, Bp, Ep, jp, Hp) = split(z_prev)

# helicityhu ic sp test
#u1 = -sin(pi*(x-1/2))*cos(pi*(y-1/2))*z0*(z0-1)
#u2 = cos(pi*(x-1/2))*sin(pi*(y-1/2))*z0*(z0-1)
#u_init = as_vector([u1, u2, 0])
##B1 = -sin(pi*x)*cos(pi*y)
#B2 = cos(pi*x)*sin(pi*y)
#B_init = as_vector([B1, B2, 0])

# ABC flow
A0 = Constant(1)
B0 = Constant(1)
C0 = Constant(1)
u1 = A0 * sin(z0) + C0 * cos(y)
u2 = B0 * sin(x) + A0 * cos(z0)
u3 = C0 * sin(y) + B0 * cos(x)

u_init = as_vector([u1, u2, u3])
B_init = as_vector([u1, u2, u3])

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

u_b_init = u_b_solver(u_init)

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

#z_prev.sub(0).interpolate(u_init)
#z_prev.sub(4).interpolate(B_init)

z_prev.sub(0).interpolate(u_init)
z_prev.sub(4).interpolate(project_ic(B_init))  # B component
z_prev.sub(2).interpolate(u_b_init)
z.assign(z_prev)

u_avg = (u + up)/2
B_avg = (B + Bp)/2
u_b_avg = u_b
P_avg = P
j_avg = j
H_avg = H
w_avg = w
E_avg = E


def spectrum_and_save(u, B, tval,
                      save_dir="output",
                      aggregate_filename="spectrum_all.csv",
                      per_timestep_files=True):
    """
    计算谱并保存为 CSV。
    - u, B: Function (2D vector) at current time
    - tval: 当前时间（float）
    - save_dir: 输出目录
    - aggregate_filename: 所有时间并成一张长表时的文件名（在 save_dir 下）
    - per_timestep_files: 是否为每一步写单独文件
    返回：k (1..kmax-1), E_u[1:], E_B[1:]
    """
    # 只让主进程写文件（调用处应保证仅 rank 0 调用或在这里再判定）
    rank = mesh.comm.rank
    if rank != 0:
        return None

    N = baseN
    x = np.linspace(0, 2 * np.pi, N, endpoint = False)
    y = np.linspace(0, 2 * np.pi, N, endpoint = False)
    z = np.linspace(0, 2 * np.pi, N, endpoint = False)

    # uniform mesh for evaluation
    u_vals = np.zeros((N, N, N, 3))
    B_vals = np.zeros((N, N, N, 3))

    for i in range(N):
        xi = x[i]
        for j in range(N):
            yj = y[j]
            for k in range(N):
                zk = z[k]
                u_vals[i, j, k, :] = u.at([xi, yj, zk])
                B_vals[i, j, k, :] = B.at([xi, yj, zk])
    
    # --- FIXED: compute 3D FFTs for each component (use full 3D slices) ---
    uhat_x = np.fft.fftn(u_vals[:, :, :, 0])
    uhat_y = np.fft.fftn(u_vals[:, :, :, 1])
    uhat_z = np.fft.fftn(u_vals[:, :, :, 2])

    Bhat_x = np.fft.fftn(B_vals[:, :, :, 0])
    Bhat_y = np.fft.fftn(B_vals[:, :, :, 1])
    Bhat_z = np.fft.fftn(B_vals[:, :, :, 2])

    # wave-number grid for domain [0, 2*pi]^3
    # spacing is d = 2*pi/N, so fftfreq(N, d=2*pi/N) and *2*pi gives angular wavenumbers
    kx = np.fft.fftfreq(N, d=2*np.pi/N) * 2*np.pi
    ky = np.fft.fftfreq(N, d=2*np.pi/N) * 2*np.pi
    kz = np.fft.fftfreq(N, d=2*np.pi/N) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # energy per Fourier mode (note: no Parseval normalization applied here,
    # keep same convention as your original code)
    E_u_k = 0.5 * (np.abs(uhat_x)**2 + np.abs(uhat_y)**2 + np.abs(uhat_z)**2)
    E_B_k = 0.5 * (np.abs(Bhat_x)**2 + np.abs(Bhat_y)**2 + np.abs(Bhat_z)**2)

    # maximum integer shell index
    kmax = int(np.max(K))
    E_u = np.zeros(kmax+1)   # make length kmax+1 so indexes 0..kmax valid
    E_B = np.zeros(kmax+1)

    # sum energy in integer-width radial shells [k, k+1)
    for kk in range(kmax+1):
        mask = (K >= kk) & (K < kk+1)
        # mask and E_u_k now have same shape (N,N,N) so indexing is valid
        E_u[kk] = np.sum(E_u_k[mask])
        E_B[kk] = np.sum(E_B_k[mask])

    # k 从 1 开始起意义更强（跳过 k=0 的总量）
    k_arr = np.arange(1, len(E_u))

    # prepare output dir
    os.makedirs(save_dir, exist_ok=True)

    # 每时间步单文件（可选择关闭）
    if per_timestep_files:
        fname = os.path.join(save_dir, f"spectrum_t={tval:.6f}.csv")
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["k", "E_u", "E_B", "E_total", "t"])
            for kk in k_arr:
                writer.writerow([int(kk), float(E_u[kk]), float(E_B[kk]), float(E_u[kk] + E_B[kk]), float(tval)])

    # 追加到汇总长表（按行存 t,k,E_u,E_B）
    agg_path = os.path.join(save_dir, aggregate_filename)
    file_exists = os.path.exists(agg_path)
    with open(agg_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["t", "k", "E_u", "E_B", "E_total"])
        for kk in k_arr:
            writer.writerow([float(tval), int(kk), float(E_u[kk]), float(E_B[kk]), float(E_u[kk] + E_B[kk])])

    # 返回便于进一步处理
    return k_arr, E_u[1:], E_B[1:]

F = (
    # u
     inner((u - up)/dt, ut) * dx
    #+ inner(filter_term(u_avg, u_b), ut) * dx # correction term
    - inner(cross(u_b_avg, w_avg), ut) * dx # advection term
    + inner(grad(P_avg), ut) * dx
    + nu * inner(curl(u_avg), curl(ut)) * dx
    - S * inner(cross(j_avg, H_avg), ut) * dx
    # p
    + inner(u_avg, grad(Pt)) * dx
    # u_b
    + inner(u_b_avg, u_bt) * dx
    + alpha**2 * inner(curl(u_b_avg), curl(u_bt)) * dx
    - inner(u_avg, u_bt) * dx
    # w
    + inner(w_avg, wt) * dx
    - inner(curl(u_avg), wt) * dx
    # B
    + inner((B - Bp)/dt, Bt) * dx
    + inner(curl(E_avg), Bt) * dx
    # E
    + inner(E_avg, Et) * dx
    + inner(cross(u_b_avg, H_avg), Et) * dx
    - eta * inner(j_avg, Et) * dx
    # j 
    + inner(j_avg, jt) * dx
    - inner(B_avg, curl(jt)) * dx
    # H
    + inner(H_avg, Ht) * dx
    - inner(B_avg, Ht) * dx
)

dirichlet_ids = ("on_boundary",)
#bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]
bcs = None

(u_, P_, u_b_, w_, B_, E_, j_, H_) = z.subfunctions
u_.rename("Velocity")
P_.rename("Pressure")
u_b_.rename("filteredVelocity")
w_.rename("Vorticity")
B_.rename("MagneticField")
E_.rename("ElectricField")
j_.rename("Current")
H_.rename("HcurlMagnetic")

pvd = VTKFile("output/mhd-alpha.pvd")

def helicity_m(B):
    A = Function(Vc)
    v = TestFunction(Vc)
    F_curl  = inner(curl(A), curl(v)) * dx - inner(B, curl(v)) * dx
    sp = {  
           "ksp_type":"preonly",
           "pc_type": "lu",
           "pc_factor_mat_solver_type": "mumps",
    }
    bcs_curl = [DirichletBC(Vc, 0, "on_boundary")]
    pb_curl = NonlinearVariationalProblem(F_curl, A, bcs_curl)
    solver_curl= NonlinearVariationalSolver(pb_curl, solver_parameters = sp, options_prefix = "solver_curlcurl")
    solver_curl.solve()
    return assemble(inner(A, B)*dx)

def norm_inf(u):
    with u.dat.vec_ro as u_v:
        u_max = u_v.norm(PETSc.NormType.INFINITY)
    return u_max

def compute_ens(w, j):
    w_max=norm_inf(w)
    j_max=norm_inf(j)
    return w_max, j_max, float(w_max) + float(j_max) 

pb = NonlinearVariationalProblem(F, z)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

timestep = 0
data_filename = "output/data.csv"
fieldnames = ["t", "divu", "divB", "energy", "helicity_c", "helicity_m", "ens_total", "w_max", "j_max"]
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

u_b = u_b_solver(z_prev.sub(0))
energy = energy_uB(z_prev.sub(0),u_b, z_prev.sub(4)) #u, u_b, B
crosshelicity = helicity_c(z.sub(0), z.sub(4)) # u, u_b, B
maghelicity = helicity_m(z.sub(4)) # B
divu = div_u(z.sub(0))
divB = div_B(z.sub(4))
# monitor
w_max, j_max, ens_total = compute_ens(z.sub(2), z.sub(6)) # w, j

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
    u_b = u_b_solver(z.sub(0)) 
    energy = energy_uB(z.sub(0),u_b, z.sub(4)) #u, u_b, B
    crosshelicity = helicity_c(z.sub(0), z.sub(4)) # u, u_b, B
    maghelicity = helicity_m(z.sub(4)) # B
    divu = div_u(z.sub(0))
    divB = div_B(z.sub(4))
    # monitor
    w_max, j_max, ens_tol = compute_ens(z.sub(2), z.sub(6)) # w, j



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
    if mesh.comm.rank == 0:
        print(row)
    
    if mesh.comm.rank == 0:
        spectrum_and_save(z.sub(0), z.sub(4), float(t))  # 会写 CSV 并存图
    pvd.write(*z.subfunctions, time=float(t))
    timestep += 1
    z_prev.assign(z)


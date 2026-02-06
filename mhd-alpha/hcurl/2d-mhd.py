# 2d-mhd
# tearing mode
from firedrake import *
import csv
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

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
Lx = 3
Ly = 1
dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
mesh = PeriodicUnitSquareMesh(baseN, baseN, distribution_parameters=dp)
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

# (u, P, u_b, w, B, E, j, H)
Z = MixedFunctionSpace([Vc, Q, Vc, Q, Vd, Q, Q, Vc])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, P, u_b, w, B, E, j, H) = split(z)
(ut, Pt, u_bt, wt, Bt, Et, jt, Ht) = split(z_test)
(up, Pp, u_bp, wp, Bp, Ep, jp, Hp) = split(z_prev)

def v_grad(x):
    return as_vector([-x.dx(1), x.dx(0)])

# Biskamp-Welter-1989
phi = cos(x + 1.4) + cos(y + 0.5)
psi = cos(2 * x + 2.3) + cos(y + 4.1)

#phi = cos(x) + cos(y)
#psi = 0.5* cos(2 * x) + cos(y)

u_init = v_grad(psi)
B_init = v_grad(phi)
  
alpha = CellDiameter(mesh)
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


u_b_init = u_b_solver(u_init)
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

def filter_term(u, u_b):
    return as_vector([
        u[0] * u_b[0].dx(0) + u[1] * u_b[1].dx(0) + u[2] * u_b[2].dx(0),  # i = 0 分量
        u[0] * u_b[0].dx(1) + u[1] * u_b[1].dx(1) + u[2] * u_b[2].dx(1),  # i = 1 分量
        u[0] * u_b[0].dx(2) + u[1] * u_b[1].dx(2) + u[2] * u_b[2].dx(2),  # i = 2 分量
    ])

def spectrum(u, B):
    N = baseN
    x = np.linspace(0, 2 * np.pi, N, endpoint = False)
    y = np.linspace(0, 2 * np.pi, N, endpoint = False)
    X, Y = np.meshgrid(x, y, indexing="ij")

# uniform mesh for evaluation
    u_vals = np.zeros((N, N, 2))
    B_vals = np.zeros((N, N, 2))

    for i in range(N):
        for j in range(N):
            u_vals[i, j, :] = u.at([x[i], y[j]])
            B_vals[i, j, :] = B.at([x[i], y[j]])

    uhat_x = np.fft.fftn(u_vals[:, :, 0])
    uhat_y = np.fft.fftn(u_vals[:, :, 1])

    Bhat_x = np.fft.fftn(B_vals[:, :, 0])
    Bhat_y = np.fft.fftn(B_vals[:, :, 1])


    kx = np.fft.fftfreq(N, d=2*np.pi/N) * 2*np.pi
    ky = np.fft.fftfreq(N, d=2*np.pi/N) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)


    E_u_k = 0.5 * (np.abs(uhat_x)**2 + np.abs(uhat_y)**2)
    E_B_k = 0.5 * (np.abs(Bhat_x)**2 + np.abs(Bhat_y)**2)


    kmax = int(np.max(K))
    E_u = np.zeros(kmax)
    E_B = np.zeros(kmax)

    for k in range(kmax):
        mask = (K >= k) & (K < k+1)
        E_u[k] = np.sum(E_u_k[mask])
        E_B[k] = np.sum(E_B_k[mask])


    k = np.arange(1, len(E_u))

    plt.figure()
    plt.loglog(k, E_u[1:], '-', label='Kinetic')
    plt.loglog(k, E_B[1:], '-.', label='Magnetic')

    # 参考 k^{-5/3}
    plt.loglog(k, 1e-2 * k**(-5/3), '--', label=r'$k^{-5/3}$')
    #plt.loglog(k, 1e-3 * k**(-3.0),  ':', label=r'$k^{-3}$')
    
    plt.xlabel(r'$k$')
    plt.ylabel(r'$E(k)$')
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("spectrum.png", dpi=300)
    plt.close()  

F = (
    # u
     inner((u - up)/dt, ut) * dx
    #+ inner(filter_term(u_avg, u_b), ut) * dx # correction term
    - inner(vcross(u_b_avg, w_avg), ut) * dx # advection term
    + inner(grad(P_avg), ut) * dx
    + nu * inner(curl(u_avg), curl(ut)) * dx
    + S * inner(vcross(H_avg, j_avg), ut) * dx
    # p
    + inner(u_avg, grad(Pt)) * dx
    # u_b
    + inner(u_b_avg, u_bt) * dx
    + alpha**2 * inner(curl(u_b_avg), curl(u_bt)) * dx
    - inner(u_avg, u_bt) * dx
    # w
    + inner(w_avg, wt) * dx
    - inner(scurl(u_avg), wt) * dx
    # B
    + inner((B - Bp)/dt, Bt) * dx
    + inner(vcurl(E_avg), Bt) * dx
    # E
    + inner(E_avg, Et) * dx
    + inner(scross(u_b_avg, H_avg), Et) * dx
    - eta * inner(j_avg, Et) * dx
    # j 
    + inner(j_avg, jt) * dx
    - inner(B_avg, vcurl(jt)) * dx
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
    # 2D is trivially 0
    return float(0)

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
        if timestep == 10:
            spectrum(z.sub(0), z.sub(4))

    pvd.write(*z.subfunctions, time=float(t))
    timestep += 1
    z_prev.assign(z)


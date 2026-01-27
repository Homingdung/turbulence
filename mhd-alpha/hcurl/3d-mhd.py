# reproduce Linshiz-Titi-2006, ref to helicityhu
# helicity, cross helicity, energy
from firedrake import *
import csv

nu = Constant(0)
eta = Constant(0)
S = Constant(1)

def helicity_c(u, B):
    return assemble(inner(u, B)*dx)

def energy_uB(u, u_b, B):
    return 0.5 * assemble(inner(u, u_b) * dx + S * inner(B, B) * dx)
    
def div_u(u):
    return norm(div(u), "L2")

def div_B(B):
    return norm(div(B), "L2")


# solver parameter
lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
sp = lu

# spatial parameters
baseN = 4
nref = 0
mesh = UnitCubeMesh(baseN, baseN, baseN)
#mesh.coordinates.dat.data[:] *= 2 * pi
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
# (u, P, u_b, lmbda, w, B, E, j, H)
Z = MixedFunctionSpace([Vc, Q, Vc, Vc, Vd, Vc, Vc, Vc])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, P, u_b, w, B, E, j, H) = split(z)
(ut, Pt, u_bt, wt, Bt, Et, jt, Ht) = split(z_test)
(up, Pp, u_bp, wp, Bp, Ep, jp, Hp) = split(z_prev)

# helicityhu ic
u1 = -sin(pi*(x-1/2))*cos(pi*(y-1/2))*z0*(z0-1)
u2 = cos(pi*(x-1/2))*sin(pi*(y-1/2))*z0*(z0-1)
u_init = as_vector([u1, u2, 0])
B1 = -sin(pi*x)*cos(pi*y)
B2 = cos(pi*x)*sin(pi*y)
B_init = as_vector([B1, B2, 0])

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

z_prev.sub(0).interpolate(u_init)
z_prev.sub(4).interpolate(B_init)

#z_prev.sub(0).interpolate(u_init)
#z_prev.sub(4).interpolate(project_ic(B_init))  # B component
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

F = (
    # u
     inner((u - up)/dt, ut) * dx
    #+ inner(filter_term(u_avg, u_b), ut) * dx # correction term
    + inner(cross(u_b_avg, w_avg), ut) * dx # advection term
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
    + inner(cross(u_b_avg, H), Et) * dx
    - eta * inner(j_avg, Et) * dx
    # j 
    + inner(j_avg, jt) * dx
    - inner(B_avg, curl(jt)) * dx
    # H
    + inner(H_avg, Ht) * dx
    + inner(B_avg, Ht) * dx
)

dirichlet_ids = ("on_boundary",)
bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids
]
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
           "ksp_type":"gmres",
           "pc_type": "ilu",
    }
    bcs_curl = [DirichletBC(Vc, 0, "on_boundary")]
    pb_curl = NonlinearVariationalProblem(F_curl, A, bcs_curl)
    solver_curl= NonlinearVariationalSolver(pb_curl, solver_parameters = sp, options_prefix = "solver_curlcurl")
    solver_curl.solve()
    return assemble(inner(A, B)*dx)

pb = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

timestep = 0
data_filename = "output/data.csv"
fieldnames = ["t", "divu", "divB", "energy", "helicity_c", "helicity_m"]

if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

energy = energy_uB(z.sub(0),z.sub(2), z.sub(4)) #u, u_b, B
crosshelicity = helicity_c(z.sub(0), z.sub(4)) # u, u_b, B
maghelicity = helicity_m(z.sub(4)) # B
divu = div_u(z.sub(0))
divB = div_B(z.sub(4))

if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "divu": float(divu),
        "divB": float(divB),
        "energy": float(energy),
        "helicity_c": float(crosshelicity),
        "helicity_m": float(maghelicity),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    if mesh.comm.rank == 0:
        print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt)}, T = {T}, baseN = {baseN}, nref = {nref}, nu = {float(nu)}", flush=True)
    solver.solve()
    
    energy = energy_uB(z.sub(0),z.sub(2), z.sub(4)) #u, u_b, B
    crosshelicity = helicity_c(z.sub(0), z.sub(4)) # u, u_b, B
    maghelicity = helicity_m(z.sub(4)) # B
    divu = div_u(z.sub(0))
    divB = div_B(z.sub(4))


    if mesh.comm.rank == 0:
        row = {
        "t": float(t),
        "divu": float(divu),
        "divB": float(divB),
        "energy": float(energy),
        "helicity_c": float(crosshelicity),
        "helicity_m": float(maghelicity),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    print(row) 
    pvd.write(*z.subfunctions, time=float(t))
    timestep += 1
    z_prev.assign(z)


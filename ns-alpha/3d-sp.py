# reproduce Miles-Rebholz discretization about NS alpha model
# helicity-conservation test
from firedrake import *
import csv

def helicity_u(u):
    return assemble(inner(u, curl(u))*dx)

def energy_u(u, u_b):
    return 0.5 * assemble(inner(u, u_b) * dx)
    
def div_u(u):
    return norm(div(u), "L2")

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

# temporal parameters
mesh = PeriodicUnitCubeMesh(baseN, baseN, baseN)
x, y, z0 = SpatialCoordinate(mesh)

# spatial discretization
Vg = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# time 
t = Constant(0) 
T = 1.0
dt = Constant(0.01)

# (u, p, u_b, lmbda, w, gamma)
Z = MixedFunctionSpace([Vg, Q, Vg, Q, Vg, Q])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, p, u_b, lmbda, w, gamma) = split(z)
(ut, pt, u_bt, lmbdat, wt, gammat) = split(z_test)
(up, pp, u_bp, lmbdap, wp, gammap) = split(z_prev)

# initial condition
nu = Constant(0)
alpha = CellSize(mesh)

u_init = as_vector([cos(2*pi*z0), sin(2*pi*z0), sin(2*pi*(x+y))])
w_init = curl(u_init)
z.sub(0).interpolate(u_init)
z.sub(4).interpolate(w_init)

z_prev.assign(z)

u_avg = (u + up)/2
u_b_avg = (u_b + u_bp)/2
w_avg = (w + wp)/2
p_avg = (p + pp)/2

F = (
    # u
      inner((u - up)/dt, ut) * dx
    -inner(cross(u_b_avg, w_avg), ut) * dx
    + inner(grad(p_avg), ut) * dx
    + nu * inner(grad(u_avg), grad(ut)) * dx
    # p
    - inner(div(u), pt) * dx
    # u_b
    + inner(u_b, u_bt) * dx
    + alpha**2 * inner(grad(u_b), grad(u_bt)) * dx
    - inner(lmbda, div(u_bt)) * dx
    - inner(u, u_bt) * dx
    # lmbda
    + inner(div(u_b), lmbdat) * dx
    # w
    + inner(curl(u), wt) * dx
    + inner(gamma, div(wt)) * dx
    - inner(w, wt) * dx
    # gamma
    + inner(div(w), gammat) * dx
)


(u_, p_, u_b_, lmbda_, w_, gamma_) = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
u_b_.rename("filteredVelocity")
lmbda_.rename("LM1")
w_.rename("Vorticity")
gamma_.rename("LM2")

pvd = VTKFile("output/3dns-alpha.pvd")
pvd.write(*z.subfunctions, time = float(t))
bcs = None
pb = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

data_filename = "output/data.csv"
fieldnames = ["t", "divu", "energy", "helicity"]

if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

energy = energy_u(z.sub(0), z.sub(2))
helicity = helicity_u(z.sub(0))
divu = div_u(z.sub(0))

if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "divu": float(divu),
        "energy": float(energy),
        "helicity": float(helicity),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

timestep = 0
while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    if mesh.comm.rank == 0:
        print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt)}, T = {T}, baseN = {baseN}, nref = {nref}, nu = {float(nu)}", flush=True)
    solver.solve()
    
    energy = energy_u(z.sub(0), z.sub(2))
    helicity = helicity_u(z.sub(0))
    divu = div_u(z.sub(0))

    if mesh.comm.rank == 0:
        row = {
            "t": float(t),
            "divu": float(divu),
            "energy": float(energy),
            "helicity": float(helicity),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    print(row) 
    pvd.write(*z.subfunctions, time=float(t))
    timestep += 1
    z_prev.assign(z)


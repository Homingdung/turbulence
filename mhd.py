# reproduce Linshiz-Titi-2006, ref to Case-LRW-2010
from firedrake import *
import csv

nu = Constant(1)
eta = Constant(1)
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
dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
baseN = 3
nref = 0

mesh = PeriodicUnitCubeMesh(baseN, baseN, baseN, distribution_parameters=dp)
mesh.coordinates.dat.data[:] *= 2 * pi
x, y, z0 = SpatialCoordinate(mesh)

# spatial discretization
Vg = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# time 
t = Constant(0) 
T = 1.0
dt = Constant(0.01)

alpha = CellDiameter(mesh)
# (u, p, u_b, r, B, lmbda)
Z = MixedFunctionSpace([Vg, Q, Vg, Q, Vg, Q])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, p, u_b, r, B, lmbda) = split(z)
(ut, pt, u_bt, rt, Bt, lmbdat) = split(z_test)
(up, pp, u_bp, rp, Bp, lmbdap) = split(z_prev)

# initial condition Politano-1995
u1 = -2 * sin(y)
u2 = 2 * sin(x)
u3 = 0
u_init = as_vector([u1, u2, 0])

B1 = 0.8 * ((-2) * sin(2*y) + sin(z0))
B2 = 0.8 * (2 * sin(x) + sin(z0))
B3 = 0.8 * (sin(x) + sin(y))
B_init = as_vector([B1, B2, B3])

z_prev.sub(0).interpolate(u_init)
z_prev.sub(4).interpolate(B_init)  # B component
z.assign(z_prev)

u_avg = (u + up)/2
p_avg = (p + pp)/2
u_b_avg = (u_b + u_bp)/2
r_avg = (r + rp)/2
B_avg = (B + Bp)/2
lmbda_avg = (lmbda + lmbdap)/2


def filter_term(u, u_b):
    return as_vector([
        u[0] * u_b[0].dx(0) + u[1] * u_b[1].dx(0) + u[2] * u_b[2].dx(0),  # i = 0 分量
        u[0] * u_b[0].dx(1) + u[1] * u_b[1].dx(1) + u[2] * u_b[2].dx(1),  # i = 1 分量
        u[0] * u_b[0].dx(2) + u[1] * u_b[1].dx(2) + u[2] * u_b[2].dx(2),  # i = 2 分量
    ])

F = (
    # u
     inner((u - up)/dt, ut) * dx
    #- inner(dot(grad(u_avg), u_b_avg), ut) * dx # advection
    - inner(cross(u_b_avg, curl(u_avg)), ut) * dx
    + inner(grad(dot(u_avg, u_b_avg)), ut) * dx
    - inner(p_avg, div(ut)) * dx
    + nu * inner(grad(u_avg), grad(ut)) * dx
    #+ inner(filter_term(u_avg, u_b), ut) * dx # correction term
    - S * inner(dot(grad(B_avg), B_avg), ut) * dx

    # p
    - inner(div(u), pt) * dx
    # u_b
    + inner(u_b_avg, u_bt) * dx
    + alpha**2 * inner(grad(u_b_avg), grad(u_bt)) * dx
    - inner(r_avg, div(u_bt)) * dx
    - inner(u_avg, u_bt) * dx
    # lmbda
    - inner(div(u_b), rt) * dx
    # B
    + inner((B - Bp)/dt, Bt) * dx
    + inner(dot(grad(B_avg), u_b_avg),  Bt) * dx
    - inner(dot(grad(u_b_avg), B_avg), Bt) * dx
    + eta * inner(grad(B_avg), grad(Bt)) * dx
    + inner(lmbda_avg, div(Bt)) * dx
    # lmbda
    + inner(div(B), lmbdat) * dx
)

bcs = None
(u_, p_, u_b_, r_, B_, lmbda_) = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
r_.rename("LM1")
B_.rename("MagneticField")
u_b_.rename("filteredVelocity")
lmbda_.rename("LM2")

pvd = VTKFile("output/mhd-alpha.pvd")
pvd.write(*z.subfunctions, time = float(t))

pb = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

timestep = 0
data_filename = "output/data.csv"
fieldnames = ["t", "divu", "divB", "energy", "helicity"]

if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

energy = energy_uB(z.sub(0),z.sub(2), z.sub(4))
helicity = helicity_c(z.sub(0), z.sub(4))
divu = div_u(z.sub(0))
divB = div_B(z.sub(4))

if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "divu": float(divu),
        "divB": float(divB),
        "energy": float(energy),
        "helicity": float(helicity),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    if mesh.comm.rank == 0:
        print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt)}, T = {T}, baseN = {baseN}, nref = {nref}, nu = {float(nu)}", flush=True)
    solver.solve()
    
    energy = energy_uB(z.sub(0),z.sub(2), z.sub(4))
    helicity = helicity_c(z.sub(0), z.sub(4))
    divu = div_u(z.sub(0))
    divB = div_B(z.sub(4))

    if mesh.comm.rank == 0:
        row = {
        "t": float(t),
        "divu": float(divu),
        "divB": float(divB),
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


# reproduce Miles-Rebholz discretization about NS alpha model
from firedrake import *
import csv

def helicity_u(u):
    return assemble(inner(u, curl(u))*dx)

def energy_u(u):
    return 0.5 * assemble(inner(u, u) * dx)
    
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
dp={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
baseN = 4
nref = 0

# temporal parameters
base = UnitCubeMesh(baseN, baseN, baseN, distribution_parameters = dp)
mh = MeshHierarchy(base, nref, distribution_parameters = dp)
# shift from [0, 1]^3 to [-1, 1]^3
scale = 2
shift = -1
# 2 * x - 1
for m in mh:
    m.coordinates.dat.data[:] = scale * m.coordinates.dat.data + shift
mesh = mh[-1]
x, y, z0 = SpatialCoordinate(mesh)

# spatial discretization
k = 2
Vg = VectorFunctionSpace(mesh, "CG", k)
Q = FunctionSpace(mesh, "CG", k-1)

# time 
t = Constant(0) 
T = 1.0
dt = Constant(0.0025)

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
alpha = CellDiameter(mesh)

a = Constant(1.25)  
d = Constant(1.0)  

damping = exp(-nu * d**2 * t)

u1_expr = -a * (exp(a * x) * sin(a * y + d * z0) + exp(a * z0) * cos(a * x + d * y)) * damping
u2_expr = -a * (exp(a * y) * sin(a * z0 + d * x) + exp(a * x) * cos(a * y + d * z0)) * damping
u3_expr = -a * (exp(a * z0) * sin(a * x + d * y) + exp(a * y) * cos(a * z0 + d * x)) * damping

term1 = exp(2*a*x) + exp(2*a*y) + exp(2*a*z0) + 2*sin(a*x+d*y)*cos(a*z0+d*x)*exp(a*(y+z0))
term2 = 2 * sin(a*y+d*z0) * cos(a*x+d*y) * exp(a*(z0+x)) * 2 * sin(a*z0+d*x) * cos(a*y+d*z0) * exp(a*(x+y))

p_init = -a**2/2 * (term1 + term2) * damping
u_init = as_vector([u1_expr, u2_expr, u3_expr])
#w_init = curl(u_init)

z_prev.sub(0).interpolate(u_init)
z_prev.sub(1).interpolate(p_init)
#z_prev.sub(4).interpolate(w_init)

z.assign(z_prev)

u_avg = (u + up)/2
u_b_avg = (u_b + u_bp)/2
w_avg = (w + wp)/2
p_avg = (p + pp)/2

F = (
    # u
    inner((u - up)/dt, ut) * dx
    -inner(cross(u_b_avg, w_avg), ut) * dx
    - inner(p_avg, div(ut)) * dx
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
    + inner(div(w), gammat) * dx
)

dirichlet_ids = ("on_boundary",)
bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]

(u_, p_, u_b_, lmbda_, w_, gamma_) = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
u_b_.rename("filteredVelocity")
lmbda_.rename("LM1")
w_.rename("Vorticity")
gamma_.rename("LM2")

pvd = VTKFile("output/3dns-alpha.pvd")
pvd.write(*z.subfunctions, time = float(t))

pb = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

data_filename = "output/data.csv"
fieldnames = ["t", "divu", "energy", "helicity"]

if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

energy = energy_u(z.sub(0))
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
    
    energy = energy_u(z.sub(0))
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


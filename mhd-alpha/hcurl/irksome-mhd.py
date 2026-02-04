# rewrite the turbulence model to fix the non-conservation of energy
# use irksome to use implicit Midpoint rule
from firedrake import *
import csv
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper

# solver parameter
lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# time paramter
t = Constant(0)
dt = Constant(0.1)
T = 1.0
butcher_tableau = GaussLegendre(1)
# physical paramater
nu = Constant(0)
eta = Constant(0)
S = Constant(1)

# mesh paramter
baseN = 4
mesh = PeriodicUnitCubeMesh(baseN, baseN, baseN)
(x, y, z0) = SpatialCoordinate(mesh)

Vd = FunctionSpace(mesh, "RT", 1)
Vc = FunctionSpace(mesh, "N1curl", 1)
Vg = FunctionSpace(mesh, "CG", 1)

#(u, P, u_b, w, B, E, j, H)
Z = MixedFunctionSpace([Vc, Vg, Vc, Vc, Vd, Vc, Vc, Vc])
z = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)

(u, P, u_b, w, B, E, j, H) = split(z)
(up, Pp, u_bp, wp, Bp, Ep, jp, Hp) = split(z_prev)
(ut, Pt, u_bt, wt, Bt, Et, jt, Ht) = split(z_test)

# initial condition
u_init = as_vector([cos(2*pi * z0), sin(2*pi*z0), sin(2*pi*x)])
B_init = as_vector([cos(2*pi * z0), sin(2*pi*z0), sin(2*pi*x)])


alpha = CellDiameter(mesh)
# solve for u_b_init
def u_b_solver(u):
    u_init = Function(Vc).interpolate(u)
    u_b = TrialFunction(Vc)
    u_sol = Function(Vc)
    v = TestFunction(Vc)
    a = inner(u_b, v) * dx + alpha**2 * inner(curl(u_b), curl(v)) * dx
    L = inner(u_init, v) * dx
    sp_ub = {  
           "ksp_type":"gmres",
           "pc_type": "ilu",
    }
    bcs0 = [DirichletBC(Vc, 0, "on_boundary")]
    pb0 = LinearVariationalProblem(a, L, u_sol, bcs = bcs0)
    solver0 = LinearVariationalSolver(pb0, solver_parameters = sp_ub)
    solver0.solve()
    return u_sol

u_b_init = u_b_solver(u_init)

z.sub(0).interpolate(u_init)
z.sub(2).interpolate(u_b_init) # important, otherwise the energy will be oscilatory
z.sub(4).interpolate(B_init)

F = (
    # u
      inner(Dt(u), ut) * dx
    - inner(cross(u_b, w), ut) * dx
    + nu * inner(curl(u), curl(ut)) * dx
    + inner(grad(P), ut) * dx
    - S * inner(cross(j, H), ut) * dx
    # - inner(f, ut) * dx
    # P
    + inner(u, grad(Pt)) * dx
    # u_b
    + inner(u_b, u_bt) * dx
    + alpha**2 * inner(curl(u_b), curl(u_bt)) *dx
    - inner(u, u_bt) * dx
    # w
    + inner(w, wt) * dx
    - inner(curl(u), wt) * dx
    #B
    + inner(Dt(B), Bt) * dx
    + inner(curl(E), Bt) * dx
    #E
    + inner(E, Et) * dx
    + inner(cross(u_b, H), Et) * dx
    - eta * inner(j, Et) * dx
    #j
    + inner(j, jt) * dx
    - inner(B, curl(jt)) * dx
    # H
    + inner(H, Ht) * dx
    - inner(B, Ht) * dx

)

pb = NonlinearVariationalProblem(F, z)
#solver = NonlinearVariationalSolver(pb, solver_parameters = lu) 
stepper = TimeStepper(F, butcher_tableau, t, dt, z, solver_parameters = lu)

def compute_div(x):
    return norm(div(x), "L2")

def compute_energy(u, u_b, B):
    return 0.5 * assemble(inner(u, u_b)*dx + S * inner(B, B) * dx)

def compute_chelicity(u, B):
    return assemble(inner(u, B) * dx)

def compute_mhelicity(B):
    A = Function(Vc)
    v = TestFunction(Vc)
    F_curl  = inner(curl(A), curl(v)) * dx - inner(B, curl(v)) * dx
    sp_curl = {  
           "ksp_type":"gmres",
           "pc_type": "ilu",
    }
    bcs = [DirichletBC(Vc, 0, "on_boundary")]
    pb_curl = NonlinearVariationalProblem(F_curl, A)
    solver_curl = NonlinearVariationalSolver(pb_curl, solver_parameters = sp_curl)
    solver_curl.solve()
    return assemble(inner(A, B)*dx)

divB = compute_div(z.sub(4)) # B
energy = compute_energy(z.sub(0), z.sub(2), z.sub(4))# u, u_b, B
helicity_c = compute_chelicity(z.sub(0), z.sub(4)) # u, B
helicity_m = compute_mhelicity(z.sub(4)) # B 

pvd = VTKFile("output/mhd-3d.pvd")
pvd.write(*z.subfunctions, time = float(t))

data_filename = "output/data.csv"
fieldnames = ["t", "divB", "energy", "helicity_m", "helicity_c"]
if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "divB": float(divB),
        "energy": float(energy),
        "helicity_c": float(helicity_c),
        "helicity_m": float(helicity_m),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
#solver.solve()
    stepper.advance()
    pvd.write(*z.subfunctions, time=float(t))
    divB = compute_div(z.sub(4)) # B
    energy = compute_energy(z.sub(0), z.sub(2), z.sub(4))# u, u_b, B
    helicity_c = compute_chelicity(z.sub(0), z.sub(4)) # u, B
    helicity_m = compute_mhelicity(z.sub(4)) # B 

    if mesh.comm.rank == 0:
        row = {
        "t": float(t),
        "divB": float(divB),
        "energy": float(energy),
        "helicity_c": float(helicity_c),
        "helicity_m": float(helicity_m),
        }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
    print(row)

    #z_prev.assign(z)



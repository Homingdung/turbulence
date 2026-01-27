# 2d mhd problem
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
mesh = UnitSquareMesh(baseN, baseN)
#mesh.coordinates.dat.data[:] *= 2 * pi
x, y = SpatialCoordinate(mesh)

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
Z = MixedFunctionSpace([Vc, Q, Vc, Q, Vd, Q, Q, Vc])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, P, u_b, w, B, E, j, H) = split(z)
(ut, Pt, u_bt, wt, Bt, Et, jt, Ht) = split(z_test)
(up, Pp, u_bp, wp, Bp, Ep, jp, Hp) = split(z_prev)

# Biskamp-Welter-1989
phi = cos(x + 1.4) + cos(y + 0.5)
psi = cos(2 * x + 2.3) + cos(y + 4.1)
def v_grad(x):
    return as_vector([-x.dx(1), x.dx(0)])

u_init = v_grad(psi)
B_init = v_grad(phi)

z_prev.sub(0).interpolate(u_init)
z_prev.sub(4).interpolate(B_init)

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
    + inner(vcross(u_b_avg, w_avg), ut) * dx # advection term
    + inner(grad(P_avg), ut) * dx
    + nu * inner(scurl(u_avg), scurl(ut)) * dx
    + S * inner(vcross(H_avg, j_avg), ut) * dx
    # p
    + inner(u_avg, grad(Pt)) * dx
    # u_b
    + inner(u_b_avg, u_bt) * dx
    + alpha**2 * inner(scurl(u_b_avg), scurl(u_bt)) * dx
    - inner(u_avg, u_bt) * dx
    # w
    + inner(w_avg, wt) * dx
    - inner(scurl(u_avg), wt) * dx
    # B
    + inner((B - Bp)/dt, Bt) * dx
    + inner(vcurl(E_avg), Bt) * dx
    # E
    + inner(E_avg, Et) * dx
    + inner(scross(u_b_avg, H), Et) * dx
    - eta * inner(j_avg, Et) * dx
    # j 
    + inner(j_avg, jt) * dx
    - inner(B_avg, vcurl(jt)) * dx
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

    return float(0)

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


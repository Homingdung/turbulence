# reproduce Miles-Rebholz discretization about NS alpha model
# ns - 2d
# remove the filter to test domain set up
from firedrake import *
import csv
import firedrake.utils as firedrake_utils

def helicity_u(u):
    w_z = scurl(u)
    w = as_vector([0, 0, w_z])
    u = as_vector([u[0], u[1], 0])
    return assemble(inner(u, w)*dx)

def energy_u(u):
    return 0.5 * assemble(inner(u, u) * dx)
    
def div_u(u):
    return norm(div(u), "L2")

def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]


def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])


def scurl(x):
    return x[1].dx(0) - x[0].dx(1)


def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])

class FixAtPointBC(firedrake.DirichletBC):
    """
    A special BC object for pinning a function at a point.
    """
    def __init__(self, V, g, bc_point):
        super(FixAtPointBC, self).__init__(V, g, bc_point)
        self.bc_point = bc_point

    @firedrake_utils.cached_property
    def nodes(self):
        V = self.function_space()
        x = firedrake.SpatialCoordinate(V.mesh())
        xdist = x - self.bc_point

        test = firedrake.TestFunction(V)
        trial = firedrake.TrialFunction(V)
        xphi = firedrake.assemble(ufl.inner(xdist * test, xdist * trial) * ufl.dx, diagonal=True)
        phi = firedrake.assemble(ufl.inner(test, trial) * ufl.dx, diagonal=True)
        with xphi.dat.vec as xu, phi.dat.vec as u:
            xu.pointwiseDivide(xu, u)
            min_index, min_value = xu.min()

        nodes = V.dof_dset.lgmap.applyInverse([min_index])
        nodes = nodes[nodes >= 0]
        return nodes

    def eval_at_point(self, V_func):
        if self.min_index is None:
            self.nodes      # To ensure that self.min_index is correctly initialized
        V_func_value = V_func.vector().gather(global_indices=[self.min_index])[0]
        return V_func_value

    def assert_is_enforced(self, V_func):
        bc_value = self.function_arg(self.bc_point)
        V_func_value = self.eval_at_point(V_func)

        assert abs(bc_value - V_func_value) < 1e-8

    def dof_index_in_mixed_space(self, M, l):
        x_sc = SpatialCoordinate(M.mesh())
        dist_func = interpolate(Constant(-1.0) + sqrt(dot(x_sc - self.bc_point, x_sc - self.bc_point)), M.sub(l))

        func_mixed = Function(M)
        func_mixed.subfunctions[l].assign(dist_func)

        with func_mixed.dat.vec as v:
            v.shift(1.0)
            min_index, min_value = v.min()
            assert min_value < 1e-8

        return min_index

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
baseN = 64
nref = 1

mesh = Mesh("mesh/bfs-2d.msh")
mh = MeshHierarchy(mesh, nref)
mesh = mh[0]
x, y = SpatialCoordinate(mesh)

# spatial discretization
k = 2
Vg = VectorFunctionSpace(mesh, "CG", k)
Vg_ = FunctionSpace(mesh, "CG", k)
Q = FunctionSpace(mesh, "CG", k-1)

# time 
t = Constant(0) 
T = 1.0
dt = Constant(0.0025)

# (u, p, u_b, lmbda, w, gamma)
Z = MixedFunctionSpace([Vg, Q])
z = Function(Z)
z_test = TestFunction(Z)
z_prev = Function(Z)

(u, p) = split(z)
(ut, pt) = split(z_test)
(up, pp) = split(z_prev)

# initial condition
nu = Constant(1e-6)
alpha = CellDiameter(mesh)

u_init = as_vector([4*(2-y)*(y-1), 0])
#z_prev.sub(0).interpolate(u_init)
#z.assign(z_prev)

u_avg = (u + up)/2
eps = lambda x: sym(grad(x))
F = (
    # u
     inner((u - up)/dt, ut) * dx
    -inner(vcross(u_avg, curl(u_avg)), ut) * dx
    - inner(p, div(ut)) * dx
    + 2 * nu * inner(eps(u_avg), eps(ut)) * dx
    # p
    - inner(div(u), pt) * dx
)

bcs = [DirichletBC(Z.sub(0), u_init, (10,)),
       DirichletBC(Z.sub(0), Constant((0,0)),(11,)),
        #DirichletBC(Z.sub(0).sub(0), Constant(1/3),(12,)), # outflow to balance the flux
        FixAtPointBC(Z.sub(1), Constant(0), as_vector([10, 1])),

]

(u_, p_) = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")

pvd = VTKFile("output/2dns-alpha.pvd")
#pvd.write(*z.subfunctions, time = float(t))

pb = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

timestep = 0
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


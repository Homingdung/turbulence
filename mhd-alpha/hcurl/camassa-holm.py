# Camassa-Holm equation
# https://www.firedrakeproject.org/demos/camassaholm.py.html

from firedrake import *
# solver parameter
lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")


alpha = 1.0
dt = Constant(0.1)
t = Constant(0)
T = 1.0
baseN = 100
mesh = PeriodicIntervalMesh(baseN, 40.0) # length 40

Vg = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([Vg, Vg])

z = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)
# v, u
(v, u) = split(z)
(vt, ut) = split(z_test)
(vp, up) = split(z_prev)

# initial condition
x, = SpatialCoordinate(mesh)
u_init = 0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.))

z_prev.sub(1).interpolate(u_init)
z.assign(z_prev)

u_avg = (u + up)/2
v_avg = (v + vp)/2

F = (
    # v
      inner((v-vp)/dt, vt) * dx
    + inner(u_avg * v_avg.dx(0), vt) * dx
    + 2 * inner(v_avg * u_avg.dx(0), vt) * dx
    # u
    + inner(v_avg, ut) * dx
    - inner(u_avg, ut) * dx
    - alpha**2 * inner(u_avg.dx(0), ut) * dx
)

pb = NonlinearVariationalProblem(F, z)
solver = NonlinearVariationalSolver(pb, solver_parameters = lu)

def energy(u):
    return assemble(inner(u, u) * dx + alpha**2 * inner(u.dx(0), u.dx(0)) * dx) 

# store the solution
pvd = VTKFile("output/ch.pvd")
pvd.write(*z.subfunctions, time=float(t))
all_sol = []

while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    print(f"energy: {energy(z.sub(1))}")
    z_prev.assign(z)
    pvd.write(*z.subfunctions, time = float(t))
    all_sol.append(z.sub(1))

try:
  from firedrake.pyplot import plot
  fig, axes = plt.subplots()
  plot(all_sol[-1], axes=axes)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)
try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)

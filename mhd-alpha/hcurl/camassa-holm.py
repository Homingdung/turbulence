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

def v_solver(u_init):
    v_b = TrialFunction(Vg)
    v_bt = TestFunction(Vg)
    v_bs = Function(Vg)
    #u_init = Function(Vg).interpolate(u)
    a = inner(v_b, v_bt) * dx
    L = inner(u_init, v_bt) * dx + alpha **2 * inner(u_init.dx(0), v_bt.dx(0)) * dx
    lu_v={
      'ksp_type': 'preonly',
      'pc_type': 'lu'
    }
    pb=LinearVariationalProblem(a, L, v_bs)
    solver = LinearVariationalSolver(pb, solver_parameters = lu_v)
    solver.solve()
    return v_bs


v_init = v_solver(u_init) # deconvolute the unfiltered velocity
z_prev.sub(0).interpolate(v_init)
z_prev.sub(1).interpolate(u_init)

z.assign(z_prev) # good for Newton's iteration

u_avg = (u + up)/2
#u_avg = u
v_avg = (v + vp)/2
F = (
    # v
      inner((v-vp)/dt, vt) * dx
    + inner(u_avg * v_avg.dx(0), vt) * dx
    + 2 * inner(v_avg * u_avg.dx(0), vt) * dx
    # u
    + inner(v, ut) * dx
    - inner(u, ut) * dx
    - alpha**2 * inner(u.dx(0), ut.dx(0)) * dx # the constraint does not use midpoint
)

pb = NonlinearVariationalProblem(F, z)
solver = NonlinearVariationalSolver(pb, solver_parameters = lu)

# use filtered velocity to compute the energy
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

#try:
#  from firedrake.pyplot import plot
#  fig, axes = plt.subplots()
#  plot(all_sol[-1], axes=axes)
#except Exception as e:
#  warning("Cannot plot figure. Error msg: '%s'" % e)
#try:
#  plt.show()
#except Exception as e:
#  warning("Cannot show figure. Error msg: '%s'" % e)

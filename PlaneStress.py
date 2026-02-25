import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
#import calfem.vis_mpl as cfv
cfv=None
import calfem.utils as cfu
import tabulate as tb
import json, math, sys
import pyvtk as vtk 
import plotly.graph_objects as go

class ModelParams:
    """Class defining the model parameters for the plane stress problem."""
    
    def __init__(self):
        self.version = 2
        self.E = 2.08e9   # Young's modulus in Pascals
        self.nu = 0.2      # Poisson's ratio
        self.t = 0.15      # Thickness in meters
        self.w = 0.3       # Width of the model in meters
        self.h = 0.1       # Height of the model in meters
        self.a = 0.05      # Width of the cutout in meters
        self.b = 0.025     # Height of the cutout in meters
        self.q = 100000    # Distributed load in Newtons per meter
        self.el_size_factor = 0.2 # Max size of generated elements
        self.param_b = False # Flag to vary b
        self.param_t = False # Flag to vary t
        self.b_end = 0.04 # End value for b
        self.t_end = 0.3 # End value for t
        self.param_filename = "param_study"
        self.param_steps = 10 # Number of steps in parameter study

    def geometry(self):
        """Create a geometry instance based on defined parameters."""
        g = cfg.Geometry()


        w = self.w # Simplyfy calling them
        h = self.h
        b = self.b
        a = self.a

        #Getting all of the important surface points

        g.point([0, 0]) #Point 1
        g.point([(w-a)*0.5, 0]) #Point 2
        g.point([(w-a)*0.5, b]) #Point 3
        g.point([(w+a)*0.5, b]) #Point 4
        g.point([(w+a)*0.5, 0]) #Point 5
        g.point([w, 0]) #Point 6
        g.point([w, h]) #Point 7
        g.point([(w+a)*0.5, h]) #Point 8
        g.point([(w+a)*0.5, h-b]) #Point 9
        g.point([(w-a)*0.5, h-b]) #Point 10
        g.point([(w-a)*0.5, h]) #Point 11
        g.point([0, h]) #Point 12

        #Create splines for the surface

        g.spline([0, 1])
        g.spline([1, 2])
        g.spline([2, 3])
        g.spline([3, 4])
        g.spline([4, 5])
        g.spline([5, 6], marker=1) #loaded
        g.spline([6, 7])
        g.spline([7, 8])
        g.spline([8, 9])
        g.spline([9, 10])
        g.spline([10, 11])
        g.spline([11, 0], marker=7) #fixed

        g.surface([0,1,2,3,4,5,6,7,8,9,10,11])

        return g

    # def save(self, filename='model_params.json'):
    #     """Save model parameters to a JSON file."""
    #     with open(filename, 'w') as file:
    #         json.dump({
    #             'version': self.version,
    #             'E': self.E,
    #             'nu': self.nu,
    #             't': self.t,
    #             'w': self.w,
    #             'h': self.h,
    #             'a': self.a,
    #             'b': self.b,
    #             'load_value': self.q,
    #             'el_size_factor': self.el_size_factor,
    #         }, file, indent=4)

    # def load(self, filename='model_params.json'):
    #     """Load model parameters from a JSON file."""
    #     with open(filename, 'r') as file:
    #         params = json.load(file)        
    #         self.version = params['version']
    #         self.E = params['E']
    #         self.nu = params['nu']
    #         self.t = params['t']
    #         self.w = params['w']
    #         self.h = params['h']
    #         self.a = params['a']
    #         self.b = params['b']
    #         self.q = params['load_value']
    #         self.el_size_factor = params['el_size_factor']



class ModelResult:
    """Class for storing results from calculations."""
    def __init__(self):
        self.a = None
        self.r = None
        self.ed = None
        self.qs = None
        self.qt = None
        self.vonMises = None
        self.coords = None
        self.edof = None
        self.bdofs = None


class ModelSolver:
    """Class for performing the model computations."""

    def __init__(self, model_params, model_result):
        """
        Initialise the solver with model parameters and a result storage object.
        
        Parameters:
            model_params (ModelParams): The parameters of the model.
            model_result (ModelResult): The object where results are stored.
        """
        self.model_params = model_params
        self.model_result = model_result

    def execute(self):
        """
        Perform the finite element analysis, solving for displacements, reactions, and other results.
        """
        # Simplifying references to model parameters
        geometry = self.model_params.geometry()        

        # Mesh generation setup
        mesh = cfm.GmshMesh(geometry)
        mesh.el_type = 2  # 
        mesh.dofs_per_node = 2  # for plane stress (u, v displacements)
        mesh.el_size_factor = self.model_params.el_size_factor  # Controls mesh density
        mesh.return_boundary_elements = True
        
        # Create the mesh
        coords, edof, dofs, bdofs, element_markers, boundary_elements = mesh.create()

        
        # Storing mesh data for visualization and further processing
        self.model_result.coords = coords
        self.model_result.edof = edof
        self.model_result.bdofs = bdofs
        self.model_result.topo = mesh.topo

        # Material properties
        D = cfc.hooke(1, self.model_params.E, self.model_params.nu)  # Plane stress
        
        ep = [1 ,self.model_params.t]  # Element properties [plane stress/strain flag, thickness]
        # Initialize global stiffness matrix and force vector
        K = np.zeros((2*len(coords), 2*len(coords)))
        f = np.zeros((2*len(coords),1))
        ex,ey = cfc.coordxtr(edof,coords,dofs)
        # Assembly of global stiffness matrix
        for elx, ely, eltopo in zip(ex, ey, edof):
            Ke = cfc.plante(elx, ely, ep, D)
            # print(Ke)
            cfc.assem(eltopo, K, Ke)

        bc= np.array([], 'i')
        bcVal = np.array([], 'f')
        load = [self.model_params.q, 0]
        # Apply loads using markers
        bc, bcVal = cfu.apply_bc(bdofs,bc,bcVal,7,0.0,0)
        #cfu.apply_force_total(bdofs, f, 1, self.model_params.q, 2)  # Apply as distributed load
        cfu.apply_traction_linear_element(boundary_elements,coords,dofs,f,1,load)
        # Apply boundary conditions using markers
        #bcPrescr = np.array(bdofs[0]).flatten()
        #bcVal = np.zeros(len(bcPrescr))

        a, r = cfc.solveq(K, f, bc, bcVal)

        # Store results
        self.model_result.a = a
        self.model_result.r = r

        # Calculate and store additional results
        ed = cfc.extract_ed(edof, a)
        self.model_result.ed = ed

        #es = []
        #et = []
        #vonMises = []

        es= np.zeros([len(edof),3])
        et= np.zeros([len(edof),3])
        vonMises= np.zeros([len(edof)])

        i=0

        for eltopo, eld, elx, ely, ees, eet, eevm in zip(edof, ed, ex, ey, es, et, vonMises):
            #ex, ey = coords[eltopo[:4],0], coords[eltopo[:4],1]
            #stress, strain = cfc.plants(elx, ely, ep, D, eld)
            #es.append(stress)
            #et.append(strain)
            # vonMises.append(cfc.effmises(stress,1))
            ies, iet = cfc.plants(elx, ely, ep, D, eld)
            ees[:] = ies[0,:]
            eet[:] = iet[0,:]
            vonMises[i]=math.sqrt(ies[0,0]**2 - ies[0,0]*ies[0,1] + ies[0,1]**2 + 3*ies[0,2]**2)
            i+=1
        #for i in range(edof.shape[0]):
        #    es,et = cfc.planqs(elx[i,:],ely[i,:],ep,D,eltopo[i,:])
        #    vonMises.append(np.sqrt(pow(es[0],2)-es[0]*es[1]+pow(es[1],2)+3*es[2]))
        
        self.model_result.es = np.array(es)
        self.model_result.et = np.array(et)
        self.model_result.vonMises = np.array(vonMises)
    
    def execute_param_study(self):
        old_b = self.model_params.b
        old_t = self.model_params.t
        results = []

        if self.model_params.param_b:
            b_range = np.linspace(self.model_params.b, self.model_params.b_end, self.model_params.param_steps)
            for i,b in enumerate(b_range):
                self.model_params.b = b
                self.execute()
                self.export_vtk(f"param_study_{i+1:02d}.vtk")
        elif self.model_params.param_t:
            t_range = np.linspace(self.model_params.t, self.model_params.t_end, self.model_params.param_steps)
            for i,t in enumerate(t_range):
                self.model_params.t = t
                self.execute()
                self.export_vtk(f"param_study_{i+1:02d}.vtk")

        self.model_params.b = old_b
        self.model_params.t = old_t
    
    def export_vtk(self, filename):
        """Export results to VTK"""
        print("Exporting results to %s." % filename)

        points = self.model_result.coords.tolist()
        polygons = (self.model_result.topo-1).tolist()

        displ = np.reshape(self.model_result.a, (len(points),2)).tolist()

        point_data = vtk.PointData(vtk.Vectors(displ, name="displacements"))


        von_mises = self.model_result.vonMises.tolist()
        cell_data = vtk.CellData(
            vtk.Scalars(von_mises, name="vonMises")
        )
        
        structure = vtk.PolyData(points=points, polygons=polygons)
        vtk_data = vtk.VtkData(structure, point_data, cell_data)
        vtk_data.tofile(filename, "ascii")

class ModelReport:
    """Class for presenting input and output parameters in report form."""
    def __init__(self, model_params, model_result):
        self.model_params = model_params
        self.model_result = model_result

    def clear(self):
        """Clear the current report content."""
        self.report = ""

    def add_text(self, text=""):
        """Add text to the report."""
        self.report += str(text) + "\n"

    def generate_report(self):
        """Generate the full report of the model inputs and outputs."""
        self.clear()

        self.add_text("-------------------------------------------------------------")
        self.add_text("-------------- Model input ----------------------------------")
        self.add_text("-------------------------------------------------------------")

        # Input parameters
        self.add_text("Model parameters:\n")
        params_data = [
            ["w [m]", self.model_params.w],
            ["h [m]", self.model_params.h],
            ["a [m]", self.model_params.a],
            ["b [m]", self.model_params.b],
            ["E [Pa]", self.model_params.E],
            ["v [-]", self.model_params.nu],
            ["t [-]", self.model_params.t],
        ]
        self.add_text(tb.tabulate(params_data, headers=["Parameter", "Value"], tablefmt="pipe"))

        # Boundary conditions
        self.add_text("\nModel boundary conditions:\n")
        bc_data = [[bc, 0] for bc in self.model_result.bdofs[7]]  # Fixed boundary markers
        self.add_text(tb.tabulate(bc_data, headers=["Nodes", "Displacement"], tablefmt="pipe"))


        # Loads
        self.add_text("\nLoads:\n")
        load_data = [["loaded", self.model_params.q, 0]]
        self.add_text(tb.tabulate(load_data, headers=["Marker", "qx[N/m]", "qy [N/m]"], tablefmt="pipe"))

        self.add_text("-------------------------------------------------------------")
        self.add_text("-------------- Results --------------------------------------")
        self.add_text("-------------------------------------------------------------")

        # General model info
        model_info_data = [
            ["Max element size:", self.model_params.el_size_factor],
            ["Dofs per node:", 2],
            ["Element type:", 2],
            ["Number of dofs:", len(self.model_result.a)],
            ["Number of elements:", len(self.model_result.edof)],
            ["Number of nodes:", len(self.model_result.coords)],
        ]
        self.add_text(tb.tabulate(model_info_data, headers=["", ""], tablefmt="pipe"))

        # Summary results
        self.add_text("\nSummary of results:\n")
        summary_data = [
            ["Max displacement:", np.max(self.model_result.a), "[m]"],
            ["Min displacement:", np.min(self.model_result.a), "[m]"],
            ["Max reaction force:", np.max(self.model_result.r), "[N]"],
            ["Min reaction force:", np.min(self.model_result.r), "[N]"],
            ["Sum reaction forces:", np.sum(self.model_result.r), "[N]"],
            ["Max vonMises:", np.max(self.model_result.vonMises), "[Pa]"],
            ["Min vonMises:", np.min(self.model_result.vonMises), "[Pa]"],
        ]
        self.add_text(tb.tabulate(summary_data, headers=["", "Value", "Unit"], tablefmt="pipe"))

        # Results per node
        self.add_text("\nResults per node:\n")
        node_data = [
            [i + 1, 2*i + 1, 2*i + 2, coord[0], coord[1], a_x, a_y, r_x, r_y]
            for i, (coord, a_x, a_y, r_x, r_y) in enumerate(
                zip(
                    self.model_result.coords,
                    self.model_result.a[::2],
                    self.model_result.a[1::2],
                    self.model_result.r[::2],
                    self.model_result.r[1::2],
                )
            )
        ]
        self.add_text(tb.tabulate(node_data, headers=["N", "dof_x", "dof_y", "x_coord [m]", "y_coord [m]", "a_x [m]", "a_y [m]", "R_x [N]", "R_y [N]"], tablefmt="pipe", floatfmt=".4f"))

        # Results per element
        element_data = [
            [i + 1, edof[1]/2, edof[3]/2, edof[5]/2, sig_xx, sig_yy, tau_xy, vMises, eps_xx, eps_yy, eps_xy]
            for i, (edof, sig_xx, sig_yy, tau_xy, vMises, eps_xx, eps_yy, eps_xy) in enumerate(
                zip(
                    self.model_result.edof,
                    self.model_result.es[:,0],
                    self.model_result.es[:,1],
                    self.model_result.es[:,2],
                    self.model_result.vonMises,
                    self.model_result.et[:,0],
                    self.model_result.et[:,1],
                    self.model_result.et[:,2],
                )
            )
        ]
        #element_data = np.zeros([len(self.model_result.edof),11])
        #for i in range(len(self.model_result.edof)):
        #    element_data[i,0]=i+1
        #    element_data[i,1]=self.model_result.edof[i,1]/2
        #    element_data[i,2]=self.model_result.edof[i,3]/2
        #    element_data[i,3]=self.model_result.edof[i,5]/2

        self.add_text("\nResult per element:\n")
        self.add_text(tb.tabulate(element_data, headers=["El", "N1", "N2", "N3", "sig_xx [Pa]", "sig_yy [Pa]", "tau_xy [Pa]", "vMise [Pa]", "eps_xx [-]", "eps_yy [-]", "eps_xy [-]"], tablefmt="psql",)) #floatfmt=".2f"))

    def __str__(self):
        self.generate_report()
        return self.report

class ModelVisualization:
    """Class for visualizing the model using CALFEM visualization tools."""
    
    def __init__(self, model_params, model_result):
        self.model_params = model_params
        self.model_result = model_result
    
    def fig_geometry(self):
        """Return a Plotly figure showing the geometry outline."""
        w = self.model_params.w
        h = self.model_params.h
        a = self.model_params.a
        b = self.model_params.b

        pts = np.array([
            [0, 0],
            [(w-a)*0.5, 0],
            [(w-a)*0.5, b],
            [(w+a)*0.5, b],
            [(w+a)*0.5, 0],
            [w, 0],
            [w, h],
            [(w+a)*0.5, h],
            [(w+a)*0.5, h-b],
            [(w-a)*0.5, h-b],
            [(w-a)*0.5, h],
            [0, h],
            [0, 0],  # close loop
        ], dtype=float)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers"))
        fig.update_layout(
            title="Geometry",
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            yaxis_scaleanchor="x",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    def fig_mesh(self):
        """Return a Plotly figure of the mesh wireframe."""
        if self.model_result.coords is None or self.model_result.topo is None:
            raise ValueError("No mesh available. Run solver first.")

        coords = self.model_result.coords
        topo = self.model_result.topo  # usually 1-based node indices

        xs, ys = [], []
        for tri in topo:
            idx = np.array(tri, dtype=int) - 1  # to 0-based
            p = coords[idx, :]
            loop = np.vstack([p, p[0]])
            xs += loop[:, 0].tolist() + [None]
            ys += loop[:, 1].tolist() + [None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines"))
        fig.update_layout(
            title="Mesh",
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            yaxis_scaleanchor="x",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    def fig_displacements(self, magnfac=50.0):
        """Return a Plotly figure of undeformed + deformed mesh wireframe."""
        if self.model_result.coords is None or self.model_result.topo is None or self.model_result.a is None:
            raise ValueError("No results available. Run solver first.")

        coords = self.model_result.coords
        topo = self.model_result.topo
        a = self.model_result.a

        disp = np.reshape(a, (len(coords), 2))
        deformed = coords + magnfac * disp

        def mesh_lines(c):
            xs, ys = [], []
            for tri in topo:
                idx = np.array(tri, dtype=int) - 1
                p = c[idx, :]
                loop = np.vstack([p, p[0]])
                xs += loop[:, 0].tolist() + [None]
                ys += loop[:, 1].tolist() + [None]
            return xs, ys

        xs0, ys0 = mesh_lines(coords)
        xs1, ys1 = mesh_lines(deformed)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs0, y=ys0, mode="lines", name="Undeformed"))
        fig.add_trace(go.Scatter(x=xs1, y=ys1, mode="lines", name=f"Deformed (x{magnfac:g})"))
        fig.update_layout(
            title="Displacements",
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            yaxis_scaleanchor="x",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    def fig_element_values(self):
        """Return a Plotly figure with vonMises values."""
        if self.model_result.coords is None or self.model_result.topo is None or self.model_result.vonMises is None:
            raise ValueError("No element values available. Run solver first.")

        coords = self.model_result.coords
        topo = self.model_result.topo
        vm = self.model_result.vonMises

        tri = np.asarray(topo, dtype=int) - 1
        if tri.shape[1] != 3:
            raise ValueError(f"Expected triangular topology (n x 3), got shape {tri.shape}")

        x = coords[:, 0]
        y = coords[:, 1]
        z = np.zeros(len(coords))

        i = tri[:, 0]
        j = tri[:, 1]
        k = tri[:, 2]

        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        dx = xmax - xmin
        dy = ymax - ymin
        pad = 0.02 * max(dx, dy) if max(dx, dy) > 0 else 1e-3

        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    intensity=vm,                 # one value per triangle
                    intensitymode="cell",         # color per element (filled)
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="vonMises [Pa]"),
                    flatshading=True,
                    opacity=1.0,
                )
            ]
        )

        # Make it look 2D
        fig.update_layout(
            title="Element Values (vonMises)",
            margin=dict(l=20, r=20, t=40, b=20),
            scene=dict(
                xaxis=dict(visible=False, range=[xmin - pad, xmax + pad]),
                yaxis=dict(visible=False, range=[ymin - pad, ymax + pad]),
                zaxis=dict(visible=False, range=[-1e-9, 1e-9]),
                aspectmode="manual",
                aspectratio=dict(
                    x=1,
                    y=(dy / dx) if dx > 0 else 1,
                    z=1e-6
                ),
                camera=dict(
                    projection=dict(type="orthographic"),
                    up=dict(x=0, y=1, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=0, z=0.15),  # straight down
                ),
            ),
        )

        return fig

    # def close_all(self):
    #     """Close all open visualization windows and reset figure references."""
    #     if self.geom_fig:

    #         self.geom_fig = None
    #     if self.mesh_fig:

    #         self.mesh_fig = None
    #     if self.nodal_values_fig:

    #         self.nodal_values_fig = None
    #     if self.element_values_fig:

    #         self.element_values_fig = None

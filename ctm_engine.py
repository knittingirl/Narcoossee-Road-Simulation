import numpy as np


class CTMEngine:
    def __init__(self, geometry, fd: 'FundamentalDiagram', dt: float):
        self.geo = geometry
        self.fd = fd
        self.dt = dt
        self.n = geometry.n_cells  # densities in veh/m per lane aggregated across lanes
        self.rho = np.zeros(self.n)  # veh/m (per lane)
        self.cell_length = geometry.cell_length_m

    def initialize_density(self, rho_init):
        rho_init = np.asarray(rho_init)
        assert rho_init.shape[0] == self.n
        self.rho = rho_init

    def step(self, external_inflow=0.0, receiving_multiplier=None, downstream_capacity=None):
        """
        One CTM time step with conservative fluxes.

        external_inflow: vehicles entering at upstream boundary (veh/s)
        receiving_multiplier: optional array of multipliers (0-1) applied to receiving of each cell
        downstream_capacity: optional cap (veh/s) on outflow from the last CTM cell.
            This is used to couple the corridor to an explicit downstream stop-line queue.
        """
        S = self.fd.sending(self.rho)
        R = self.fd.receiving(self.rho)
        if receiving_multiplier is not None:
            R = R * receiving_multiplier

        flux = np.zeros(self.n + 1)
        flux[0] = min(external_inflow, R[0])
        for i in range(self.n - 1):
            flux[i + 1] = min(S[i], R[i + 1])

        last_cell_outflow = S[-1]
        if downstream_capacity is not None:
            last_cell_outflow = min(last_cell_outflow, downstream_capacity)
        flux[self.n] = last_cell_outflow

        inflow = flux[:-1]
        outflow = flux[1:]
        delta = (self.dt / self.cell_length) * (inflow - outflow)
        self.rho = np.maximum(self.rho + delta, 0.0)
        return flux

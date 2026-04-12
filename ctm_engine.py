import numpy as np

class CTMEngine: 
  def init(self, geometry, fd: 'FundamentalDiagram', dt: float): 
    self.geo = geometry 
    self.fd = fd 
    self.dt = dt 
    self.n = geometry.n_cells # densities in veh/m per lane aggregated across lanes 
    self.rho = np.zeros(self.n) # veh/m (per lane) 
    # convert storage to densities if needed 
    self.cell_length = geometry.cell_length_m

def initialize_density(self, rho_init):
    rho_init = np.asarray(rho_init)
    assert rho_init.shape[0] == self.n
    self.rho = rho_init

def step(self, external_inflow=0.0, receiving_multiplier=None):
    """
    One CTM time step with conservative fluxes.
    external_inflow: vehicles entering at upstream boundary (veh/s)
    receiving_multiplier: optional array of multipliers (0-1) applied to receiving of each cell
    """
    S = self.fd.sending(self.rho)  # veh/s (per lane aggregated by lanes inside FD)
    R = self.fd.receiving(self.rho)
    if receiving_multiplier is not None:
        R = R * receiving_multiplier

    # fluxes between i -> i+1
    flux = np.zeros(self.n + 1)  # flux[0] is upstream boundary, flux[n] is outflow
    # upstream boundary limited by upstream sending and first cell receiving
    flux[0] = min(external_inflow, R[0])
    for i in range(self.n - 1):
        flux[i+1] = min(S[i], R[i+1])
    # outflow from last cell is its sending
    flux[self.n] = S[-1]

    # update densities conservatively: rho_new = rho + (dt/length) * (inflow - outflow)
    inflow = flux[:-1]
    outflow = flux[1:]
    delta = (self.dt / self.cell_length) * (inflow - outflow)
    self.rho = np.maximum(self.rho + delta, 0.0)
    return flux

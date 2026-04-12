import numpy as np from dataclasses import dataclass

@dataclass 
class FundamentalDiagram: 
  v_free: float # free-flow speed (m/s) 
  rho_crit: float # critical density (veh/m) 
  rho_jam: float # jam density (veh/m) 
  lanes: int = 1

  def flow_capacity(self):
      return self.v_free * self.rho_crit * self.lanes
  
  def sending(self, rho):
      """
      Sending function S(rho) = min(v_free * rho, capacity)
      rho: density in veh/m (can be array)
      """
      return np.minimum(self.v_free * rho * self.lanes, self.flow_capacity())
  
  def receiving(self, rho):
      """
      Receiving function R(rho) = w * (rho_jam - rho)
      where w is backward wave speed computed from FD parameters
      """
      # compute backward wave speed w from triangular FD: w = q_max / (rho_jam - rho_crit)
      q_max = self.flow_capacity()
      w = q_max / (self.rho_jam - self.rho_crit + 1e-9)
      return w * (self.rho_jam - rho) * self.lanes

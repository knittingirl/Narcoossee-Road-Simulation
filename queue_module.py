import numpy as np

class StopLineQueue: 
  def init(self, saturation_flow_veh_per_s: float): 
    self.q = 0.0 
    self.saturation_flow = saturation_flow_veh_per_s

  def update(self, arrivals, green_fraction, dt):
      """
      Discrete queue update at time resolution dt
      q(t+dt) = q(t) + arrivals*dt - departures
      departures limited by green_fraction * saturation_flow * dt
      arrivals: veh/s entering the queue during dt
      green_fraction: fraction of cycle that is green (0-1) during this dt
      """
      arrivals_veh = arrivals * dt
      max_departures = green_fraction * self.saturation_flow * dt
      departures = min(self.q + arrivals_veh, max_departures)
      self.q = max(self.q + arrivals_veh - departures, 0.0)
      return departures

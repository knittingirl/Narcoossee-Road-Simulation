import numpy as np

class PlatoonArrivalGenerator: 
  def init(self, horizon_s, dt, base_rate_veh_per_s, burst_prob=0.1, mean_burst_size=5, intra_burst_headway_s=1.0, seed=None): 
    self.horizon_s = horizon_s 
    self.dt = dt 
    self.steps = int(np.ceil(horizon_s / dt)) 
    self.base_rate = base_rate_veh_per_s 
    self.burst_prob = burst_prob 
    self.mean_burst_size = mean_burst_size 
    self.intra_burst_headway = intra_burst_headway_s 
    self.rng = np.random.default_rng(seed)

  def generate(self):
      """
      Returns arrivals array of length steps with arrival rate (veh/s) per dt interval.
      Mixes Poisson baseline with occasional bursts (platoons).
      """
      rates = np.full(self.steps, self.base_rate)
      # add time-varying modulation (example: simple peak in middle)
      t = np.arange(self.steps) * self.dt
      peak = 1.0 + 0.5 * np.exp(-((t - self.horizon_s/2)**2) / (2*(self.horizon_s/6)**2))
      rates *= peak
  
      # baseline Poisson arrivals converted to rate per second already represented by rates
      arrivals = np.zeros(self.steps)
      for i in range(self.steps):
          # baseline Poisson count in dt
          lam = rates[i] * self.dt
          baseline_count = self.rng.poisson(lam)
          arrivals[i] += baseline_count / self.dt  # convert back to veh/s for consistency
  
          # burst generation
          if self.rng.random() < self.burst_prob:
              burst_size = max(1, self.rng.poisson(self.mean_burst_size))
              # place burst vehicles across subsequent time steps according to intra-burst headway
              for k in range(burst_size):
                  arrival_time = i * self.dt + k * self.intra_burst_headway
                  idx = int(np.floor(arrival_time / self.dt))
                  if idx < self.steps:
                      arrivals[idx] += 1.0 / self.dt
      return arrivals

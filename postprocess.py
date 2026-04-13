import numpy as np 
import pandas as pd

def compute_mean_queue(queue_time_series): 
  arr = np.asarray(queue_time_series) 
  return arr.mean()

def travel_time_stats(travel_times_s): 
  arr = np.asarray(travel_times_s) 
  return {"mean": arr.mean(), "p50": np.percentile(arr, 50), "p90": np.percentile(arr, 90)}

def spillback_frequency(spillback_flags): 
  return np.sum(spillback_flags)

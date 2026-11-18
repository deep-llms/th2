"""Bounded, balanced four-stage curricula; no stochastic policy or future inputs."""
from .model import ORDERS

def path_for(arm,step,total):
 if total%80 or total<80 or step<0 or step>=total:raise ValueError('Schedule needs positive complete80-update cycles')
 if arm=='D':return ORDERS[0]
 if arm=='R':return ORDERS[step%8]
 if arm not in ('WS','WL'):raise ValueError('Unknown curriculum')
 boundaries=[total//10,2*total//10,3*total//10] if arm=='WS' else [total//4,total//2,3*total//4]
 depth=1+sum(step>=b for b in boundaries)
 return ORDERS[step%8][:depth]

# CityLearn – EV Model + RL-MPC + Thermal Models
CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. A major challenge for RL in demand response is the ability to compare algorithm performance. Thus, CityLearn facilitates and standardizes the evaluation of RL agents such that different algorithms can be easily compared with each other.

![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/dr.jpg)

This repo extends CityLearn with:

RL-MPC agent (SLSQP-based MPC with an LSTM thermal forecaster)
New energy models: Fresnel solar-thermal collector, Thermal Buffer (TES), Absorption Chiller
Existing SAC agents and our EV integration

The goal is to make it easy to compare MPC/RL (and hybrids) on building coordination and demand response. MPC and RLC have a hierarchical architecture. Local LSTM-based MPC manages HVAC loads, while centralized SAC-based RL coordinates energy storage systems.

<img width="1065" height="641" alt="GraphicAbstract" src="https://github.com/user-attachments/assets/fac10530-0e09-479c-b12a-6bea44e77f17" />

## Environment Overview

CityLearn includes energy models of buildings and distributed energy resources (DER) including air-to-water heat pumps, electric heaters and batteries. A collection of building energy models makes up a virtual district (a.k.a neighborhood or community). In each building, space cooling, space heating and domestic hot water end-use loads may be independently satisfied through air-to-water heat pumps. Alternatively, space heating and domestic hot water loads can be satisfied through electric heaters.
New energy models: Fresnel solar-thermal collector, Thermal Buffer (TES), Absorption Chiller

## What’s new in this fork

agents/MPC: receding-horizon controller that solves a small optimization each step.
energy/thermal.py:
FresnelCollector → thermal kWh from irradiance
ThermalBuffer → thermal storage (losses, charge/discharge caps)
AbsorptionChiller → thermal-driven cooling (COP)
Obs/Actions added to buildings
Observations:
fresnel_thermal_output
thermal_buffer_soc
absorption_chiller_cooling_output
Actions:
fsc_to_buffer ∈ [0,1] – fraction of Fresnel output sent to buffer
buffer_to_chiller ∈ [0,1] – fraction of buffer capacity discharged to chiller this step

## Installation
Install latest release in PyPi with `pip`:
```console
!pip install git+https://github.com/ParisaAligol/CityLearn-EVModel-Parisa@develop-ev
```

## Documentation
This work builds on CityLearn by the Intelligent Environments Lab. Refer to the [docs](https://intelligent-environments-lab.github.io/CityLearn/) for documentation of the CityLearn API.
Refer to the paper "A Hierarchical Energy Management System for a Cluster of Buildings: Reinforcement Learning and Model Predictive Control (RL-MPC) Approach"

##Repo layout
.
├─ our_package/
│  ├─ agents/
│  │  ├─ sac.py
│  │  └─ mpc.py
│  ├─ energy/
│  │  └─ thermal.py            # FresnelCollector, ThermalBuffer, AbsorptionChiller
│  ├─ building.py              # Building with new devices/actions/obs
│  └─ models/
│     └─ lstm.py
├─ examples/
│  └─ rlmpc_with_fsc_and_abc_and_cooling_storage.py
├─ notebooks/
│  └─ MPC.ipynb
├─ pyproject.toml (or requirements.txt)
└─ README.md

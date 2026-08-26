import pyomo.environ as pyomo
import numpy as np
import pandas as pd

# Hardcoded data for a residential region
data = {
    "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # 24-hour format
    "price": [0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.04, 0.06, 0.075, 0.075, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.075, 0.075, 0.06, 0.05, 0.04, 0.035, 0.035],  # Electricity prices ($/kWh)
    "input_energy": [1000, 1000, 1000, 1000, 1000, 1000, 1200, 1500, 2000, 2000, 1500, 1200, 1200, 1200, 1200, 1500, 2000, 2000, 1500, 1200, 1000, 1000, 1000, 1000],  # Input energy (kW)
    "hourly_demand": [500, 400, 300, 300, 400, 800, 1200, 1500, 1800, 1500, 1200, 1000, 1000, 1000, 1000, 1200, 1500, 1800, 1500, 1200, 1000, 800, 600, 500],  # Hourly demand (kW)
}

# Create DataFrame
df = pd.DataFrame(data)
df.set_index("time", inplace=True)

# Extract prices and hourly demand
prices = {t: price for t, price in enumerate(df["price"])}
hourly_demand = {t: demand for t, demand in enumerate(df["hourly_demand"])}

# Create Pyomo model
model = pyomo.ConcreteModel()

# Parameters
model.T = pyomo.Set(initialize=range(len(prices)))  # Time steps

# Storage parameters
model.soc_init = pyomo.Param(initialize=1000.0)  # Initial state of charge (kWh)
model.soc_max = pyomo.Param(initialize=10000.0)  # Increased maximum state of charge (kWh)
model.soc_min = pyomo.Param(initialize=0.0)  # Minimum state of charge (kWh)
model.sell_max = pyomo.Param(initialize=1000.0)  # Increased maximum sell power (kW)
model.input_energy_max = pyomo.Param(initialize=1000.0)  # Increased maximum input energy (kW)
model.charge_rate_max = pyomo.Param(initialize=1000.0)  # Increased maximum charging rate (kW)
model.discharge_rate_max = pyomo.Param(initialize=1000.0)  # Increased maximum discharging rate (kW)

# Price and demand parameters
model.price = pyomo.Param(model.T, initialize=prices)  # Electricity prices
model.hourly_demand = pyomo.Param(model.T, initialize=hourly_demand)  # Hourly demand

# Variables
model.v_sell = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  # Power sold to the grid (kW)
model.v_soc = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  # State of charge (kWh)
model.v_input_energy = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  # Input energy (kW)
model.v_is_selling = pyomo.Var(model.T, domain=pyomo.Binary)  # Binary variable for selling
model.v_charge = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  # Charging power (kW)
model.v_discharge = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  # Discharging power (kW)

# Constraints
def sell_max_rule(model, t):
    """Limit the power sold to the grid."""
    return model.v_sell[t] <= model.v_is_selling[t] * model.sell_max

def soc_max_rule(model, t):
    """Limit the maximum state of charge."""
    return model.v_soc[t] <= model.soc_max

def soc_min_rule(model, t):
    """Limit the minimum state of charge."""
    return model.v_soc[t] >= model.soc_min

def energy_equilibrium_rule(model, t):
    """Energy balance constraint."""
    if t == 0:
        return model.soc_init + model.v_input_energy[t] - model.v_sell[t] - model.v_discharge[t] + model.v_charge[t] == model.v_soc[t]
    else:
        return model.v_soc[t - 1] + model.v_input_energy[t] - model.v_sell[t] - model.v_discharge[t] + model.v_charge[t] == model.v_soc[t]

def input_energy_max_rule(model, t):
    """Limit the input energy based on the selling status."""
    return model.v_input_energy[t] <= model.input_energy_max * (1 - model.v_is_selling[t])

def charge_rate_rule(model, t):
    """Limit the maximum charging rate."""
    return model.v_charge[t] <= model.charge_rate_max

def discharge_rate_rule(model, t):
    """Limit the maximum discharging rate."""
    return model.v_discharge[t] <= model.discharge_rate_max

def demand_satisfaction_rule(model, t):
    """Ensure the hourly demand is met."""
    return model.v_discharge[t] + model.v_input_energy[t] >= model.hourly_demand[t]

# Add constraints to the model
model.c_sell_max = pyomo.Constraint(model.T, rule=sell_max_rule)
model.c_soc_max = pyomo.Constraint(model.T, rule=soc_max_rule)
model.c_soc_min = pyomo.Constraint(model.T, rule=soc_min_rule)
model.c_energy_equilibrium = pyomo.Constraint(model.T, rule=energy_equilibrium_rule)
model.c_input_energy_max = pyomo.Constraint(model.T, rule=input_energy_max_rule)
model.c_charge_rate = pyomo.Constraint(model.T, rule=charge_rate_rule)
model.c_discharge_rate = pyomo.Constraint(model.T, rule=discharge_rate_rule)
model.c_demand_satisfaction = pyomo.Constraint(model.T, rule=demand_satisfaction_rule)

# Objective Function
def objective_func(model):
    """Maximize revenue from selling electricity."""
    return sum((model.v_sell[t] - model.v_input_energy[t]) * model.price[t] for t in model.T)

model.obj = pyomo.Objective(rule=objective_func, sense=pyomo.maximize)

# Solve the model
solver = pyomo.SolverFactory("glpk")  # Use GLPK solver
results = solver.solve(model)

# Check if the solver was successful
if results.solver.status == pyomo.SolverStatus.ok and results.solver.termination_condition == pyomo.TerminationCondition.optimal:
    print("Solver was successful!")

    # Extract results
    sell_array = np.zeros(len(prices))
    soc_array = np.zeros(len(prices))
    input_energy_array = np.zeros(len(prices))
    charge_array = np.zeros(len(prices))
    discharge_array = np.zeros(len(prices))

    for t in model.T:
        sell_array[t] = pyomo.value(model.v_sell[t])
        soc_array[t] = pyomo.value(model.v_soc[t])
        input_energy_array[t] = pyomo.value(model.v_input_energy[t])
        charge_array[t] = pyomo.value(model.v_charge[t])
        discharge_array[t] = pyomo.value(model.v_discharge[t])

    # Add results to the DataFrame
    df["soc"] = soc_array
    df["sell"] = sell_array
    df["input_energy"] = input_energy_array
    df["charge"] = charge_array
    df["discharge"] = discharge_array

    # Print results
    print("Optimization results:")
    print(df)
    print("Total revenue:", pyomo.value(model.obj))
else:
    print("Solver failed to find an optimal solution.")
    print("Solver status:", results.solver.status)
    print("Termination condition:", results.solver.termination_condition)
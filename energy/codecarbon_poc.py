from codecarbon import EmissionsTracker

import time
import math


tracker = EmissionsTracker()
tracker.start()

# Your code here
end = time.time() + 5  # run for ~5 seconds
x = 0.0001

while time.time() < end:
    x = math.sin(x) ** 2 + math.cos(x) ** 2  # pointless but CPU-heavy

emissions = tracker.stop()

# Access energy (kWh)
energy_kwh = tracker.final_emissions_data.energy_consumed

print("---------------------------------------------")

print(f"Emissions: {emissions} kg CO₂")
print(f"Energy consumed: {energy_kwh} kWh")
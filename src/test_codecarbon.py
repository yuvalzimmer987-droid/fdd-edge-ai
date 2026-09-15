from codecarbon import EmissionsTracker
import time

# Start measuring
tracker = EmissionsTracker(project_name="fdd_test", log_level="error")
tracker.start()

# A small dummy workload (just to have something to measure)
total = 0
for i in range(10_000_000):
    total += i * i
time.sleep(1)

# Stop and read the result
tracker.stop()
data = tracker.final_emissions_data
print("\n--- CodeCarbon test ---")
print(f"Energy consumed: {data.energy_consumed:.8f} kWh")
print(f"Duration:        {data.duration:.2f} seconds")
print("CodeCarbon is working." if data.energy_consumed >= 0 else "Something is off.")

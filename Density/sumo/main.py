import traci
import os
import sys

def run_simulation():
    try:
        # Path to the SUMO configuration file
        sumo_config_path = "D:/project_C/2024-10-06-11-51-19/finalC.sumocfg"

        # Check if the configuration file exists
        if not os.path.exists(sumo_config_path):
            print(f"Error: Configuration file '{sumo_config_path}' does not exist.")
            sys.exit(1)

        # Start SUMO with the specified configuration file and set the remote port for TraCI
        traci.start(["sumo", "-c", sumo_config_path, "--remote-port", "54147"])
        
        # Simulation loop
        step = 0
        while step < 1000:  # Assuming you want to run the simulation for 1000 steps
            traci.simulationStep()  # Advance the simulation by one step

            # List of traffic lights
            tls_ids = traci.trafficlight.getIDList()

            # Iterate over each traffic light
            for tls_id in tls_ids:
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                total_weighted_density = 0

                # Iterate over each lane controlled by this traffic light
                for lane in controlled_lanes:
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)

                    # Iterate over each vehicle in the lane and apply weight based on type
                    for vehicle_id in vehicle_ids:
                        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                        if vehicle_type == "car":
                            total_weighted_density += 1  # Weight for car
                        elif vehicle_type == "truck":
                            total_weighted_density += 2  # Weight for truck

                # Control traffic light duration based on the weighted density
                if total_weighted_density > 10:  # Example threshold for density
                    # Extend green phase for traffic light to reduce congestion
                    current_phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
                    traci.trafficlight.setPhaseDuration(tls_id, current_phase_duration + 10)

            # Increment simulation step
            step += 1

        # Close the TraCI connection after simulation ends
        traci.close()
    
    except traci.exceptions.FatalTraCIError as e:
        # If there's an error, print it and exit
        print(f"Fatal TraCI Error: {e}")
        sys.exit(1)

    except Exception as e:
        # Catch any other general exceptions
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_simulation()
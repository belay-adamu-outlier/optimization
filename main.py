from simulation_utils import (
    create_simulation_tasks,
    save_results_to_csv,
    create_detailed_results_csv,
    print_simulation_summary
)
from parallel_task_runner import ParallelTaskRunner, get_physical_core_count
from datetime import datetime
import os


def main():
    # Configuration
    idd_file = 'C:/EnergyPlusV9-4-0/Energy+.idd'  # Update this path
    weather_file = './Vancouver - 2021 AMY.epw'
    
    # IDF files in the current directory
    idf_files = [
        './Building.idf',
        './Building copy.idf',
        './Building - Copy.idf',
        './Building - Copy (2).idf',
        './Building - Copy (3).idf',
        './Building - Copy (4).idf',
        './Building - Copy (5).idf',
        './Building - Copy (6).idf',
        './Building - Copy (7).idf'
    ]
    
    if not idf_files:
        print("No IDF files specified")
        return
    
    print(f"Found {len(idf_files)} IDF files: {[os.path.basename(f) for f in idf_files]}")
    
    # Create simulation tasks
    tasks = create_simulation_tasks(idf_files, weather_file, idd_file)
    
    
    print(f"Using {4} workers for parallel processing")
    
    # Create parallel task runner
    runner = ParallelTaskRunner(tasks, 4)
    
    print(f"Starting parallel simulations with {len(idf_files)} files...")
    
    # Run simulations in parallel
    results = runner.run()
    
    print(f"Completed {len(results)} simulations")
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary results
    summary_csv = f'./simulation_results/summary_{timestamp}.csv'
    save_results_to_csv(results, summary_csv)
    
    # Save detailed time series results
    detailed_csv = f'./simulation_results/detailed_{timestamp}.csv'
    create_detailed_results_csv(results, detailed_csv)
    
    # Print summary
    print_simulation_summary(results)


if __name__ == '__main__':
    main()
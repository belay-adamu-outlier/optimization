#!/usr/bin/env python3
"""
Simulate envelope cases with shading from CSV file.
Filters duplicates and runs parallel EnergyPlus simulations.
"""

import os
import csv
import multiprocessing
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

from util.energyplus_utils import (
    run_simulation,
    analyze_simulation_results,
)
from constants import (
    FILE_PATHS,
    SIM_PARAMS,
    SHADING_CONFIG,
    CSV_CONFIG,
    EnvelopeCase,
    SimulationResult
)



class EnvelopeCaseSimulator:
    """Main class for simulating envelope cases with shading - simplified approach with duplicate filtering."""

    def __init__(
        self,
        csv_path: str = None,
        idf_path: str = None,
        weather_path: str = None,
        num_workers: int = None,
        output_csv: str = None
    ):
        # Use constants as defaults, but allow overrides
        self.csv_path = csv_path or FILE_PATHS.CSV_INPUT
        self.idf_path = idf_path or FILE_PATHS.IDF_TEMPLATE
        self.weather_path = weather_path or FILE_PATHS.WEATHER_FILE
        self.num_workers = num_workers or SIM_PARAMS.DEFAULT_NUM_WORKERS
        self.output_csv = output_csv or f"envelope_shading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self.shading_dimensions = SHADING_CONFIG.dimensions

        self.unique_cases: List[EnvelopeCase] = []
        self.duplicate_count = 0

        # Will collect individual result files from workers
        self.temp_result_files = []

    def read_envelope_cases(self) -> List[EnvelopeCase]:
        """Read and filter envelope cases from CSV file, removing duplicates."""
        envelope_groups = defaultdict(list)

        with open(self.csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if not all([row.get(col) for col in CSV_CONFIG.required_csv_columns]):
                    continue

                tedi = float(row['Thermal Energy Deman Intensity'])
                if tedi <= SIM_PARAMS.TEDI_THRESHOLD:
                    case = EnvelopeCase(
                        generation=int(row['Generation']),
                        iteration=int(row['Iteration']),
                        external_wall=row['external_wall'],
                        roof=row['roof'],
                        window=row['window'],
                        infiltration_ach=float(row['infiltration_ach']),
                        wwr_east=float(row['wwr_east']),
                        wwr_west=float(row['wwr_west']),
                        wwr_north=float(row['wwr_north']),
                        wwr_south=float(row['wwr_south']),
                        wall_solar_absorptance=float(row['wall_solar_absorptance']),
                        roof_solar_absorptance=float(row['roof_solar_absorptance']),
                        tedi=tedi,
                        max_lr_oh=float(row['Max_Living_Room_Overheating']),
                        max_br_oh=float(row['Max_Bedroom_Overheating'])
                    )

                    envelope_key = (
                        case.external_wall, case.roof, case.window,
                        case.infiltration_ach, case.wwr_east, case.wwr_west,
                        case.wwr_north, case.wwr_south,
                        case.wall_solar_absorptance, case.roof_solar_absorptance
                    )

                    envelope_groups[envelope_key].append(case)

        unique_cases = []
        total_duplicates = 0

        for envelope_key, cases_with_same_envelope in envelope_groups.items():
            unique_cases.append(cases_with_same_envelope[0])
            if len(cases_with_same_envelope) > 1:
                total_duplicates += len(cases_with_same_envelope) - 1

        self.unique_cases = unique_cases
        self.duplicate_count = total_duplicates

        total_cases_processed = len(list(csv.DictReader(open(self.csv_path, 'r'))))
        print(f"Found {len(unique_cases)} unique combinations with TEDI <= 15")
        print(f"Filtered out {total_duplicates} duplicates")
        print(f"Original dataset: {total_cases_processed} total cases")

        return unique_cases

    def prepare_envelope_params(self, case: EnvelopeCase) -> Dict[str, Any]:
        return {
            'external_wall': case.external_wall,
            'roof': case.roof,
            'window': case.window,
            'infiltration_ach': case.infiltration_ach,
            'wwr_east': case.wwr_east,
            'wwr_west': case.wwr_west,
            'wwr_north': case.wwr_north,
            'wwr_south': case.wwr_south,
            'wall_solar_absorptance': case.wall_solar_absorptance,
            'roof_solar_absorptance': case.roof_solar_absorptance,
            'shading_dimensions': self.shading_dimensions
        }

    def run_single_simulation(self, case: EnvelopeCase) -> tuple[SimulationResult, str | None]:
        """Run simulation and return result along with temp file path."""
        params = self.prepare_envelope_params(case)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        output_dir = run_simulation(
            idf_path=self.idf_path,
            weather_path=self.weather_path,
            params=params,
            idd_path=FILE_PATHS.IDD_PATH,
            output_dir=f"simulation_output_{timestamp}_pid{os.getpid()}",
            verbose='q'
        )

        results = analyze_simulation_results(
            output_dir,
            cleanup=False,  # Clean up after analysis
            overheating_criterion='CIBSE59'
        )

        # Create the result object
        result = SimulationResult(
            case=case,
            tedi_shading=results['district_heating_kwh'] / SIM_PARAMS.TEDI_CONVERSION_FACTOR if results['success'] else 0.0,
            max_bdrm_oh_shading=results.get('max_bedroom_overheating', 0.0) if results['success'] else 0.0,
            max_lr_oh_shading=results.get('max_living_room_overheating', 0.0) if results['success'] else 0.0,
            success=results['success'],
            error=None if results['success'] else results.get('error', 'Unknown error')
        )

        # Write result to temporary file
        temp_file = self._write_result_to_temp_file(result)

        return (result, temp_file)

    def _write_result_to_temp_file(self, result: SimulationResult) -> str:
        """Write result to a temporary CSV file for this process."""
        import tempfile

        # Create temp file with unique name based on process and case
        pid = os.getpid()
        temp_fd, temp_path = tempfile.mkstemp(suffix=f'_pid{pid}_gen{result.case.generation}_iter{result.case.iteration}.csv')

        try:
            with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as temp_file:
                fieldnames = CSV_CONFIG.output_fieldnames
                writer = csv.DictWriter(temp_file, fieldnames=fieldnames)

                # Write header
                writer.writeheader()

                # Write data row
                row = {
                    'Generation': result.case.generation,
                    'Iteration': result.case.iteration,
                    'external_wall': result.case.external_wall,
                    'roof': result.case.roof,
                    'window': result.case.window,
                    'infiltration_ach': result.case.infiltration_ach,
                    'wwr_east': result.case.wwr_east,
                    'wwr_west': result.case.wwr_west,
                    'wwr_north': result.case.wwr_north,
                    'wwr_south': result.case.wwr_south,
                    'wall_solar_absorptance': result.case.wall_solar_absorptance,
                    'roof_solar_absorptance': result.case.roof_solar_absorptance,
                    'TEDI': result.case.tedi,
                    'Max_LR_OH': result.case.max_lr_oh,
                    'Max_BR_OH': result.case.max_br_oh,
                    'TEDI_Shading': f"{result.tedi_shading:.4f}" if result.success else 'FAILED',
                    'Max_bdrm_OH_Shading': f"{result.max_bdrm_oh_shading:.2f}" if result.success else 'FAILED',
                    'Max_LR_OH_Shading': f"{result.max_lr_oh_shading:.2f}" if result.success else 'FAILED',
                    'Success': 'Yes' if result.success else 'No'
                }
                writer.writerow(row)

        except Exception as e:
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

        return temp_path

    def _merge_temp_files_to_main_csv(self):
        """Merge all temporary result files into the main CSV file."""
        import tempfile

        # Collect all temp files from system temp directory
        temp_files = []
        temp_dir = tempfile.gettempdir()

        # Look for files with our specific pattern
        for file in os.listdir(temp_dir):
            if file.startswith('tmp') and file.endswith('.csv') and '_pid' in file and '_gen' in file and '_iter' in file:
                temp_files.append(os.path.join(temp_dir, file))

        if not temp_files:
            return

        print(f"Merging {len(temp_files)} temporary result files...")

        # Read all temp files and collect their data rows
        all_rows = []
        for temp_file in sorted(temp_files):
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)

        # Write all rows to main CSV file
        if all_rows:
            with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                # Write metadata header
                csvfile.write('# Envelope Shading Simulation Results\n')
                metadata = {
                    'Source_File': self.csv_path,
                    'IDF_File': self.idf_path,
                    'Weather_File': self.weather_path,
                    'Num_Workers': self.num_workers,
                    'TEDI_Threshold': SIM_PARAMS.TEDI_THRESHOLD,
                    'Shading_Config': '2m overhang + fins on south/west facades',
                    'Overheating_Criterion': 'CIBSE59',
                    'Generated_At': datetime.now().isoformat()
                }
                for key, value in metadata.items():
                    csvfile.write(f'# {key}: {value}\n')
                csvfile.write('#\n')

                # Write data
                fieldnames = CSV_CONFIG.output_fieldnames
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

        print(f"Merged {len(all_rows)} results into main CSV file")


    def run_parallel_simulations(self, cases: List[EnvelopeCase]) -> List[SimulationResult]:
        print(f"Starting parallel simulation of {len(cases)} unique cases with {self.num_workers} workers...")
        print("(Each worker writes results to temp file immediately after completion)")

        with multiprocessing.Pool(self.num_workers) as pool:
            # Map returns list of (result, temp_file_path) tuples
            result_tuples = pool.map(self.run_single_simulation, cases)

        # Extract results and collect temp file paths
        results = []
        temp_files = []
        for result, temp_file in result_tuples:
            results.append(result)
            if temp_file:
                temp_files.append(temp_file)

        # Merge all temp files into main CSV
        self._merge_temp_files_to_main_csv()

        return results

    def get_simulation_statistics(self) -> Dict[str, int]:
        return {
            'unique_cases': len(self.unique_cases),
            'duplicate_cases_filtered': self.duplicate_count,
            'total_cases_processed': len(self.unique_cases) + self.duplicate_count
        }

    def initialize_output_csv(self):
        metadata = {
            'Source_File': self.csv_path,
            'IDF_File': self.idf_path,
            'Weather_File': self.weather_path,
            'Num_Workers': self.num_workers,
            'TEDI_Threshold': SIM_PARAMS.TEDI_THRESHOLD,
            'Shading_Config': '2m overhang + fins on south/west facades',
            'Overheating_Criterion': 'CIBSE59',
            'Generated_At': datetime.now().isoformat()
        }

        fieldnames = CSV_CONFIG.output_fieldnames

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write('# Envelope Shading Simulation Results\n')
            for key, value in metadata.items():
                csvfile.write(f'# {key}: {value}\n')
            csvfile.write('#\n')

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def save_results_to_csv(self, results: List[SimulationResult]):
        with open(self.output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = CSV_CONFIG.output_fieldnames
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for result in results:
                row = {
                    'Generation': result.case.generation,
                    'Iteration': result.case.iteration,
                    'external_wall': result.case.external_wall,
                    'roof': result.case.roof,
                    'window': result.case.window,
                    'infiltration_ach': result.case.infiltration_ach,
                    'wwr_east': result.case.wwr_east,
                    'wwr_west': result.case.wwr_west,
                    'wwr_north': result.case.wwr_north,
                    'wwr_south': result.case.wwr_south,
                    'wall_solar_absorptance': result.case.wall_solar_absorptance,
                    'roof_solar_absorptance': result.case.roof_solar_absorptance,
                    'TEDI': result.case.tedi,
                    'Max_LR_OH': result.case.max_lr_oh,
                    'Max_BR_OH': result.case.max_br_oh,
                    'TEDI_Shading': f"{result.tedi_shading:.4f}" if result.success else 'FAILED',
                    'Max_bdrm_OH_Shading': f"{result.max_bdrm_oh_shading:.2f}" if result.success else 'FAILED',
                    'Max_LR_OH_Shading': f"{result.max_lr_oh_shading:.2f}" if result.success else 'FAILED',
                    'Success': 'Yes' if result.success else 'No'
                }
                writer.writerow(row)

    def run(self, max_cases: int = None):
        """Run the complete simulation process with consolidated error handling."""
        try:
            print("Starting envelope case simulation with shading...")

            # Load and filter cases
            cases = self.read_envelope_cases()

            if not cases:
                print("No cases found with TEDI <= 15. Exiting.")
                return

            # Limit cases for testing (if specified)
            if max_cases and len(cases) > max_cases:
                original_count = len(cases)
                cases = cases[:max_cases]
                print(f"Limiting to {len(cases)} cases for testing (was {original_count} cases)")

            # Run parallel simulations
            results = self.run_parallel_simulations(cases)

            # Get statistics and print summary
            sim_stats = self.get_simulation_statistics()
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful

            print("\nSimulation Summary:")
            print(f"  Test cases simulated: {len(results)}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Success rate: {successful/len(results)*100:.1f}%")
            print(f"  Dataset stats - Duplicates filtered: {sim_stats['duplicate_cases_filtered']}")
            print(f"  Results saved incrementally to: {self.output_csv}")

        except Exception as e:
            print(f"Simulation failed with error: {e}")
            print("Some results may have been saved to temp files.")
            print(f"Check {self.output_csv} for any partial results.")
            raise


def main(max_cases: int = None):
    """Main function for testing with limited cases."""
    simulator = EnvelopeCaseSimulator()
    simulator.run(max_cases=max_cases)


if __name__ == "__main__":
    main()

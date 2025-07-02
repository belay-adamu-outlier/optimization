import multiprocessing
import threading
import os


def get_physical_core_count():
    """
    Attempt to determine the number of physical CPU cores (not logical processors).
    Falls back to logical CPU count if detection fails.
    """
    # Try psutil first (popular 3rd-party library)
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        if phys is not None:
            return phys
    except ImportError:
        pass
    # Fallback: on Linux use sched_getaffinity to count allowed cores
    try:
        phys = len(os.sched_getaffinity(0))
        return phys
    except AttributeError:
        pass
    # Last resort: assume no SMT, logical == physical
    return multiprocessing.cpu_count()


class ParallelTaskRunner:
    """
    A generic runner that executes a function across many scenarios in parallel,
    using callbacks to immediately fill freed worker slots without polling.
    """

    def __init__(self, func, num_workers=None):
        """
        :param func: callable that accepts one scenario and returns a result
        :param num_workers: optional int. If None, uses physical_cores.
                            Must be >=1 and <= physical_cores.
        """
        self.func = func
        physical_cores = get_physical_core_count()
        print(f"Physical cores: {physical_cores}")
        # Use all available physical cores
        default_workers = physical_cores

        if num_workers is None:
            num_workers = default_workers
        elif num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        elif num_workers > default_workers:
            # Cap to physical core count
            print(f"Warning: num_workers={num_workers} exceeds physical cores={default_workers}; capping to {default_workers}")
            num_workers = default_workers

        self.num_workers = num_workers

    def run(self, scenarios):
        """
        Execute func(scenario) for each scenario in the list, in parallel.

        :param scenarios: list of scenario objects
        :return: list of results in order of completion
        """
        total = len(scenarios)
        if total == 0:
            return []

        pending = list(scenarios)
        results = []
        completed = 0
        done_event = threading.Event()

        pool = multiprocessing.Pool(processes=self.num_workers)

        def _on_complete(result):
            nonlocal completed
            results.append(result)
            completed += 1
            # Dispatch next scenario if any remain
            if pending:
                next_scenario = pending.pop(0)
                pool.apply_async(self.func,
                                 args=(next_scenario,),
                                 callback=_on_complete)
            # Signal when all done
            if completed == total:
                done_event.set()

        # Seed initial jobs
        for _ in range(min(self.num_workers, total)):
            scenario = pending.pop(0)
            pool.apply_async(self.func,
                             args=(scenario,),
                             callback=_on_complete)

        # Wait until all callbacks signal completion
        done_event.wait()
        pool.close()
        pool.join()

        return results

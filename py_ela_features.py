import concurrent.futures
import json
import os

from src.constant import MAX_WORKERS
from src.instance.BBOB_Instance import BBOB_Instance
from src.log import logger

if __name__ == "__main__":
    function_index = int(os.environ.get("FUNCTION_INDEX").strip())
    dimension_list = [2, 3, 5, 10, 20]
    instance_index_list = [1, 2, 3, 4, 5]

    features = {}
    futures = []

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)

    for dimension in dimension_list:
        for instance_index in instance_index_list:
            instance = BBOB_Instance(
                function_index=function_index,
                dimension=dimension,
                instance_index=instance_index,
            )
            future = executor.submit(instance._calculate_features, instance)
            futures.append((instance, future))

    for instance, future in futures:
        try:
            result, time = future.result(3600)
            features[instance.id()] = {"result": result, "time": time}
            logger.info(
                f"Calculated features for instance {instance} in {time:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Error calculating features for instance {instance}: {e}")

    with open(f"features_{function_index}.json", "w") as f:
        json.dump(features, f, indent=4)

    executor.shutdown(wait=False, cancel_futures=True)

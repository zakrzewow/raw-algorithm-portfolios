from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_from_index_file
from src.log import logger

if __name__ == "__main__":
    instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json"
    )
    for instance in instances[:10]:
        logger.debug(f"calculate_features(instance={instance})")
        instance._calculate_ubc_features()

    # instances = TSP_from_index_file(filepath=DATA_DIR / "TSP" / "MY" / "index.json")
    # with ProcessPoolExecutor(max_workers=9) as executor:
    #     for instance in instances:
    #         logger.debug(f"calculate_features(instance={instance})")
    #         instance.calculate_features(executor)

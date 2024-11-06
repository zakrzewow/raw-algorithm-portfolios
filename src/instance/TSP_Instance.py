from src.instance import Instance


class TSP_Instance(Instance):
    def __init__(self, filepath: str, optimum: float):
        self.filepath = filepath
        self.optimum = optimum

from typing import Optional, List

class TestConfig:
    def __init__(self, bias_mit: str, ml_method: str, bias_ml_method: Optional[str]=None, preprocessing: Optional[str]=None, sensitive_attr:Optional[List[str]]=None, other = {}) -> None:
        self.bias_mit=bias_mit
        self.ml_method=ml_method
        self.bias_ml_method=bias_ml_method
        self.preprocessing=preprocessing
        self.sensitive_attr=sensitive_attr
        self.other=other

    def __hash__(self):
        # Use hash function on all relevant attributes
        hash_tuple = (
            self.bias_mit,
            self.ml_method,
            self.bias_ml_method,
            self.preprocessing,
            tuple(self.sensitive_attr) if self.sensitive_attr else None,
            frozenset(self.other.items()) if self.other else None,
        )
        return hash(hash_tuple)
    
    def __str__(self):
        attrs = [
            f"{key}={value}"
            for key, value in self.__dict__.items()
            if key not in ["other"]
        ]
        other_attrs = [
            f"{key}={value}"
            for key, value in self.other.items()
        ]
        return f"TestConfig({', '.join(attrs + other_attrs)})"

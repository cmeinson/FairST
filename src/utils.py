from typing import Optional, List

class TestConfig:
    
    def __init__(self, bias_mit: str, ml_method: str, bias_ml_method: Optional[str]=None, preprocessing: Optional[str]=None, base_model_bias_mit: Optional[str]=None, sensitive_attr:Optional[List[str]]=None, other = {}) -> None:
        self.bias_mit = bias_mit
        self.ml_method = ml_method
        self.bias_ml_method = bias_ml_method
        self.preprocessing = preprocessing
        self.sensitive_attr = sensitive_attr
        self.base_model_bias_mit = base_model_bias_mit # for post porc methods
        self.other=other

    def __hash__(self):
        str_other = {}
        if self.other is not None:
            str_other = {key: str(value) for key, value in self.other.items()}

        # Use hash function on all relevant attributes
        hash_tuple = (
            self.bias_mit,
            self.ml_method,
            self.bias_ml_method,
            self.preprocessing,
            self.base_model_bias_mit,
            tuple(self.sensitive_attr) if self.sensitive_attr else None,
            frozenset(str_other.items()),
        )
        return hash(hash_tuple)
    
    def get_hash_without_ml_method(self):
        # Use hash function on all relevant attributes
        str_other = {}
        if self.other is not None:
            str_other = {key: str(value) for key, value in self.other.items()}

        hash_tuple = (
            self.bias_mit,
            self.bias_ml_method,
            self.preprocessing,
            self.base_model_bias_mit,
            tuple(self.sensitive_attr) if self.sensitive_attr else None,
            frozenset(str_other.items()),
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
    
    def get_base_model_config(self):
        config = TestConfig(
            bias_mit=self.base_model_bias_mit,
            ml_method=self.ml_method,
            bias_ml_method=self.bias_ml_method,
            preprocessing=self.preprocessing,
            sensitive_attr=self.sensitive_attr,
            base_model_bias_mit = None,
            other = self.other            
        )
        return config

from ..ml_interface import Model
import pandas as pd
import numpy as np
from typing import List
import torch
import torch.optim as optim
from .models import *
from .losses import *
from ...test_config import TestConfig
from .configs import VAEMaskConfig
from collections import Counter

from torch.nn.utils import clip_grad_norm_


PRINT_FREQ = 1000
VERBOSE = False


class VAEMaskModel(Model):
    VAE_MASK_CONFIG = "my model config"

    COUNTER = 0

    def __init__(
        self, config: TestConfig, base_model: Model, post_mask_transform: callable
    ) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super().__init__(config)
        self._model = base_model
        self._mask_model = None
        self._column_names = None
        self._vm_config = config.other[self.VAE_MASK_CONFIG]
        self._mask_weights = {}
        self.post_mask_transform = post_mask_transform
        self._has_been_fitted = False

    def fit(self, X: pd.DataFrame, y: np.array):
        """Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        """
        self._column_names = X.columns.tolist()
        input_dim = len(self._column_names)
        sens_column_ids = [
            self._column_names.index(col) for col in self._config.sensitive_attr
        ]

        self._vm_config.set_input_dim_and_sens_column_ids(input_dim, sens_column_ids)
        self._set_mask_values_and_weights(X, sens_column_ids)
        print(self._mask_weights)

        # Build the mask_model
        if len(self._vm_config.lossed_used) > 0:
            tensor = torch.tensor(X.values, dtype=torch.float32)
            self._mask_model = self._train_vae(tensor, y, self._vm_config.epochs)


    def predict(self, X: pd.DataFrame, binary=True) -> np.array:
        preds = np.zeros(len(X))
        for attrs, w in self._mask_weights.items():
            X_masked = self._mask(X, list(attrs))
            preds = preds + (self._model.predict(X_masked, binary=False) * w)

        if not binary:
            return preds
        return self._binarise(preds)

    def _mask(self, X: pd.DataFrame, mask_values: List[int]) -> pd.DataFrame:
        if len(self._vm_config.lossed_used) == 0:  # if vae not used.
            return self.post_mask_transform(self._mask_only_attr(X, mask_values))

        tensor = torch.tensor(X.values, dtype=torch.float32)

        if VERBOSE:
            print("BEFORE MASK")
            print(tensor)
        self._mask_model.eval()

        tensor_out, *_ = self._mask_model.forward(tensor, mask_values, use_std=False)
        df = pd.DataFrame(tensor_out.detach().numpy(), columns=self._column_names)

        if VERBOSE:
            print("AFTER MASK")
            print(df)
            print(self.post_mask_transform(df))
        return self.post_mask_transform(df)

    def _mask_only_attr(self, X: pd.DataFrame, mask_values: List[int]) -> pd.DataFrame:
        for i, val in zip(self._vm_config.sens_column_ids, mask_values):
            print("manually setting ", X.columns[i], val)
            X.iloc[:, i] = val

        return X

    def _train_vae(self, X: Tensor, y: np.array, epochs=500):
        if self._has_been_fitted:
            return self._mask_model

        self.COUNTER = 0
        model = VAE(self._vm_config)
        optimizer = optim.Adam(
            model.parameters(), lr=self._vm_config.lr
        )  # , momentum=0.3)
        model.train()

        loss_models = [
            self._get_loss_model(name) for name in self._vm_config.lossed_used
        ]

        for epoch in range(epochs):
            self.COUNTER += 1
            # predict on main model -> mu, std, z, decoded
            outputs, mu, std, z, attr_cols = model.forward(X)

            # train each loss model and get loss
            loss = self._get_total_loss(
                loss_models, X, outputs, mu, std, z, attr_cols, model, y
            )

            optimizer.zero_grad()
            
            # train main model
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # print all used losses
            if self.COUNTER % PRINT_FREQ == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.7f}")
        model.eval()
        return model

    def _get_total_loss(
        self, loss_models, original_X, decoded_X, mu, std, z, attr_cols, vae_model, y
    ):
        losses = []
        for model in loss_models:
            model.set_current_state(
                original_X, decoded_X, mu, std, z, attr_cols, vae_model, y
            )
            losses.append(model.get_loss())
            if self.COUNTER % PRINT_FREQ == 0:
                print("------------------", model, "loss:  ", losses[-1])

        return sum(losses)

    def _get_loss_model(self, name) -> LossModel:
        classes = {
            VAEMaskConfig.KL_DIV_LOSS: KLDivergenceLoss,
            VAEMaskConfig.RECON_LOSS: ReconstructionLoss,
            VAEMaskConfig.LATENT_S_ADV_LOSS: LatentDiscrLoss,
            VAEMaskConfig.FLIPPED_ADV_LOSS: FlippedDiscrLoss,
            VAEMaskConfig.KL_SENSITIVE_LOSS: SensitiveKLLoss,
            VAEMaskConfig.POS_VECTOR_LOSS: PositiveVectorLoss,
        }

        loss_config = self._vm_config.loss_configs[name]
        return classes[name](loss_config)

    def _set_mask_values_and_weights(self, X: pd.DataFrame, sens_column_ids):
        if self._vm_config.mask_values is not None:
            if self._vm_config.mask_values != -1:
                self._mask_weights = {tuple(self._vm_config.mask_values): 1}
                return

        tensor_values = list(map(tuple, X.iloc[:, sens_column_ids].values))
        n = len(tensor_values)
        counts = dict(Counter(tensor_values))

        for attrs, count in counts.items():
            if self._vm_config.mask_values == -1:  # equal weights
                self._mask_weights[attrs] = 1 / len(attrs)

            elif count > 0:  # proportional weights
                self._mask_weights[attrs] = count / n

        print(self._mask_weights)

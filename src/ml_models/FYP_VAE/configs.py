
class VAEMaskConfig:
    KL_DIV_LOSS = "KL divergence loss"
    RECON_LOSS = "Reconstruction loss"
    LATENT_S_ADV_LOSS = "z->s ADV loss"
    FLIPPED_ADV_LOSS = "Flipped ADV loss"
    KL_SENSITIVE_LOSS = "Sensitive KL loss"
    POS_VECTOR_LOSS = "POS Y VEC loss"
    #def __init__(self, epochs = 500, latent_dim = 10, vae_layers = (90, 60, 30), losses_used = [KL_DIV_LOSS, RECON_LOSS, LATENT_S_ADV_LOSS]):
    def __init__(self, epochs = 500, latent_dim = 10, vae_layers = (75, 60, 30, 15), lr = 0.007, losses_used = [KL_DIV_LOSS, RECON_LOSS, LATENT_S_ADV_LOSS], mask_values = None):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.vae_layers = vae_layers
        self.lossed_used = losses_used
        self.lr = lr
        self.loss_configs = {}
        self.sens_column_ids = None
        self.input_dim = None
        self.non_sens_latent_dim = None
        self.mask_values = mask_values # NOTE: if none, will use all mask values and weigh proportionally to training data subgroup sizes
        for loss in losses_used:
            self.config_loss(loss)

    def config_loss(self, loss_name, **kwargs):
        if loss_name == self.KL_DIV_LOSS:
            self._config_KL_div(**kwargs)
        elif loss_name == self.RECON_LOSS:
            self._config_recon(**kwargs)
        elif loss_name == self.LATENT_S_ADV_LOSS:
            self._config_latent_s(**kwargs)
        elif loss_name == self.FLIPPED_ADV_LOSS:
            self._config_flipped(**kwargs)
        elif loss_name == self.KL_SENSITIVE_LOSS:
            self._config_KL_sens(**kwargs)
        elif loss_name == self.POS_VECTOR_LOSS:
            self._config_pos_vec(**kwargs)

    def set_input_dim_and_sens_column_ids(self, input_dim, ids):
        # TODO: this class got kinda ugly
        print("input dim:", input_dim)
        self.input_dim = input_dim
        self.sens_column_ids = ids
        self.non_sens_latent_dim = self.latent_dim - len(ids)
        #for loss_config in self.loss_configs.values():
        #    loss_config["non_sens_latent_dim"] = self.latent_dim - len(ids)

        if self.FLIPPED_ADV_LOSS in self.loss_configs:
            self.loss_configs[self.FLIPPED_ADV_LOSS]["input_dim"] = input_dim
            self.loss_configs[self.FLIPPED_ADV_LOSS]["sens_col_ids"] = ids

        if self.KL_SENSITIVE_LOSS in self.loss_configs:
            self.loss_configs[self.KL_SENSITIVE_LOSS]["sens_col_ids"] = ids
            
        if self.LATENT_S_ADV_LOSS in self.loss_configs:
            self.loss_configs[self.LATENT_S_ADV_LOSS]["input_dim"] = self.non_sens_latent_dim


    #def _config_KL_div(self, weight=0.005): valye during first successfult tests
    def _config_KL_div(self, weight=0.005):
        self.loss_configs[self.KL_DIV_LOSS] = {
            "weight": weight
        }
        print(self.loss_configs[self.KL_DIV_LOSS])

    #def _config_recon(self, weight=12):
    def _config_recon(self, weight=15):
        self.loss_configs[self.RECON_LOSS] = {
            "weight": weight
        }

    def _config_latent_s(self, weight=0.11, lr=0.05, optimizer="Adam", layers=(30,30)):
        self.loss_configs[self.LATENT_S_ADV_LOSS] = {
            "weight": weight,
            "lr": lr,
            "optimizer": optimizer,
            "layers": layers,
            "input_dim": self.non_sens_latent_dim,
        }
        print(self.loss_configs[self.LATENT_S_ADV_LOSS])

    def _config_flipped(self, weight=0.01, lr=0.05, optimizer="Adam", layers=(50,30,10)):
        self.loss_configs[self.FLIPPED_ADV_LOSS] = {
            "weight": weight,
            "lr": lr,
            "optimizer": optimizer,
            "layers": layers,
            "input_dim": self.input_dim,
            "sens_col_ids" : self.sens_column_ids
        }
        print(self.loss_configs[self.FLIPPED_ADV_LOSS])

    #def _config_KL_sens(self, weight=0.005):
    def _config_KL_sens(self, weight=9000):
        self.loss_configs[self.KL_SENSITIVE_LOSS] = {
            "weight": weight,
            "sens_col_ids" : self.sens_column_ids
        }
        print(self.loss_configs[self.KL_SENSITIVE_LOSS])

    def _config_pos_vec(self, weight=1200000):
        self.loss_configs[self.POS_VECTOR_LOSS] = {
            "weight": weight
        }

    def __str__(self):
        config_str = (
            f"VAEMaskConfig(epochs={self.epochs}, latent_dim={self.latent_dim}, mask_values={self.mask_values} "
            f"vae_layers={self.vae_layers}, lr={self.lr}, \nlosses_used={self.lossed_used})"
        )
        config_str += ", ".join([f"{name}: {config}" for name, config in self.loss_configs.items()])
        return config_str


    

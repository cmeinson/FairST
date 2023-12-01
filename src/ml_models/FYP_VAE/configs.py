
class VAEMaskConfig:
    KL_DIV_LOSS = "KL divergence loss"
    RECON_LOSS = "Reconstruction loss"
    LATENT_S_ADV_LOSS = "loss from an adversaty trying to predict z->s"
    def __init__(self, epochs = 100, latent_dim = 10, vae_layers = (90, 60, 30), losses_used = {KL_DIV_LOSS, RECON_LOSS, LATENT_S_ADV_LOSS}):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.vae_layers = vae_layers
        self.lossed_used = losses_used
        self.loss_configs = {}
        self.sens_column_ids = None
        self.input_dim = None
        for loss in losses_used:
            self.config_loss(loss)

    def config_loss(self, loss_name, *kwargs):
        if loss_name == self.KL_DIV_LOSS:
            self._config_KL_div(*kwargs)
        elif loss_name == self.RECON_LOSS:
            self._config_recon(*kwargs)
        elif loss_name == self.LATENT_S_ADV_LOSS:
            self._config_latent_s(*kwargs)

    def set_input_dim_and_sens_column_ids(self, input_dim, ids):
        self.input_dim = input_dim
        self.sens_column_ids = ids
        for loss_config in self.loss_configs.values():
            loss_config["non_sens_latent_dim"] = self.latent_dim - len(ids)

    def _config_KL_div(self, weight=0.01):
        self.loss_configs[self.KL_DIV_LOSS] = {
            "weight": weight
        }

    def _config_recon(self, weight=1):
        self.loss_configs[self.RECON_LOSS] = {
            "weight": weight
        }

    def _config_latent_s(self, weight=1, lr=0.05, optimizer="Adam", layers=(30,30)):
        self.loss_configs[self.LATENT_S_ADV_LOSS] = {
            "weight": weight,
            "lr": lr,
            "optimizer": optimizer,
            "layers": layers,
            "latent_dim": self.latent_dim
        }

    

    

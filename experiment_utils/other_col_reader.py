from src.ml_models.FYP_VAE.configs import VAEMaskConfig as vae_config  # has losses

def get_config(row, pre, post = ',', loss = None, numeric = True):
    other = row["other"]
    
    if "VAEMaskConfig" not in other:
        if numeric:
            return 0
        return '-'
    
    if loss is not None:
        other = other.split(loss)[1]
    other = other.split(pre)[1]
    res =  other.split(post)[0]
    if numeric:
        return float(res)
    return res

class OtherColReader:
    EPOCHS = "epochs"
    LATENT_DIM = "latent dimentions"
    VAE_LAYERS = "vae layers"
    LR = "learning rate"
    LOSSES = "losses used"
    
    L_W = "L: weight"
    F_W = "F: weight"
    K_W = "K: weight"
    P_W = "P: weight"
    
    L_LR = "L: lr"
    F_LR = "F: lr"
    
    L_L = "L: layers"
    F_L = "F: layers"
    
    ACC_SF_TO = "acc * sf"
    
    def __init__(self, df) -> None:
        self.df = df
    
    def add_col(self, name):
        funcs = {
           self.EPOCHS: lambda row: get_config(row, 'epochs='),
           self.LATENT_DIM: lambda row: get_config(row, 'latent_dim='),
           self.VAE_LAYERS: lambda row: get_config(row, 'vae_layers=', post='),',numeric=False),
           self.LR: lambda row: get_config(row, 'lr='),
           self.LOSSES: self.get_losses,
           
           self.L_LR: lambda row: get_config(row, 'lr=', vae_config.LATENT_S_ADV_LOSS),
           self.F_LR: lambda row: get_config(row, 'lr=', vae_config.FLIPPED_ADV_LOSS),
           
           self.ACC_SF_TO: lambda row: row["accuracy"]*row["[SF] Statistical Parity Subgroup Fairness"]
        }
        func = funcs[name]
        self.df[name] = self.df.apply(func, axis=1)   
        
        # return True if numeric
        if name in [self.VAE_LAYERS, self.LOSSES]:
            return False
        return True     
    
   

    def get_losses(self, row):
        other = row["other"]

        if "VAEMaskConfig" not in other:
            return "-"
        losses = get_config(row, 'losses_used=', post='])', numeric=False)
        
        split_losses = losses.split(',')
        out = ''
        for l in split_losses:
            name = l.split("'")[1]    
            out = out+name[0]
        return out
    
# for each loss: w, lr, layers
# main: epochs=1120, latent_dim=8, vae_layers,  lr=0.01386, type of loss ()
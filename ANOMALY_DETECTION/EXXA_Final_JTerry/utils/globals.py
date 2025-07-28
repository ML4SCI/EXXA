data_path = "./Data/Keplerian/"

# physical constants
au_km = 1.496e8
au_m = au_km * 1000.0
au_cm = au_m * 100.0
au_pc = 206265.0
watt_to_jansky = 1e-26
c_kms = 299792.0
G = 6.6743e-11
G_cgs = 6.6743e-8
Ms_kg = 1.989e30
Ms_g = Ms_kg * 1e3
MJ_g = 1.899e30
au_pc = 206265.0

# plotting parameters
scale_factor = 1.5
labels = 20 * scale_factor
ticks = 14 * scale_factor
legends = 14 * scale_factor
text = 14 * scale_factor
titles = 22 * scale_factor
lw = 3 * scale_factor
ps = 200 * scale_factor
cmap = "magma"
colors = [
    "firebrick",
    "steelblue",
    "darkorange",
    "darkviolet",
    "cyan",
    "magenta",
    "darkgreen",
    "deeppink",
]
markers = ["x", "o", "+", ">", "*", "D", "4"]
linestyles = ["-", "--", ":", "-."]

# model stuff
model_types = {}
# model_types["hardy-sweep-10"] = "autoencoder"
# model_types["autumn-star-38"] = "transformer"
# model_types["bumbling-field-39"] = "transformer"
# model_types["upbeat-yogurt-40"] = "transformer"
# model_types["eternal-sweep-4"] = "transformer"
# model_types["valiant-sweep-5"] = "transformer"
# model_types["lucky-sweep-7"] = "transformer"
# model_types["usual-deluge-25"] = "new_transformer"
# model_types["vocal-butterfly-25"] = "multiloader_transformer"
# model_types["royal-breeze-29"] = "multiloader_transformer"
# model_types["glowing-plasma-34"] = "multiloader_transformer"
# model_types["robust-sweep-6"] = "multiloader_transformer"
# model_types["fragrant-sweep-17"] = "multiloader_transformer"
# model_types["chocolate-sweep-33"] = "multiloader_autoencoder"

model_input_dims = {}
model_input_dims["autoencoder"] = "sequence"
model_input_dims["transformer"] = 1
model_input_dims["new_transformer"] = 1
model_input_dims["multiloader_transformer"] = 1

model_output_dims = {}
model_output_dims["autoencoder"] = "sequence"
model_output_dims["transformer"] = 1
model_output_dims["new_transformer"] = 1
model_output_dims["multiloader_transformer"] = 1

all_model_hparams = {}
# all_model_hparams["hardy-sweep-10"] = {
#     "activation": "gelu",
#     "lr": 0.0008979255140301339,
#     "adam_eps": 0.00004775821548690664,
#     "weight_decay": 0.00000003903602912228,
#     "latent_dim": 19,
#     "num_mlp_layers": 3,
#     "mlp_layer_dim": 127,
#     "dropout": 0.1189291566392146,
#     "input_dim": 101,
#     "line_index": 1,
#     "leaky_relu_frac": 0.2,
#     "max_seq_length": 101,
#     "max_v": 5,
#     "output_dim": 101,
#     "scale_data": -1.0,
#     "use_batchnorm": 0,
#     "weight_init": "xavier",
#     "add_noise": 0,
# }
# all_model_hparams["autumn-star-38"] = {
#     "input_dim": 1,
#     "output_dim": 1,
#     "hidden_dim": 4,
#     "num_layers": 3,
#     "num_heads": 4,
#     "pf_dim": 48,
#     "dropout": 0.2,
#     "seq_length": 75,
#     "lr": 1e-4,
#     "weight_decay": 1e-8,
#     "adam_eps": 1e-7,
#     "activation": "gelu",
#     "add_noise": False,
#     "device": "mps",
#     "trg_eq_zero": True,
#     "one_per_run": True,
#     "scale_data": -1,
# }

# all_model_hparams["bumbling-field-39"] = {
#     "input_dim": 1,
#     "output_dim": 1,
#     "hidden_dim": 4,
#     "num_layers": 3,
#     "num_heads": 4,
#     "pf_dim": 48,
#     "dropout": 0.2,
#     "seq_length": 101,
#     "lr": 1e-4,
#     "weight_decay": 1e-8,
#     "adam_eps": 1e-7,
#     "activation": "gelu",
#     "add_noise": False,
#     "device": "mps",
#     "trg_eq_zero": True,
#     "one_per_run": True,
#     "scale_data": -1,
# }

# all_model_hparams["upbeat-yogurt-40"] = {
#     "input_dim": 1,
#     "output_dim": 1,
#     "hidden_dim": 4,
#     "num_layers": 3,
#     "num_heads": 4,
#     "pf_dim": 32,
#     "dropout": 0.2,
#     "seq_length": 101,
#     "lr": 1e-4,
#     "weight_decay": 1e-8,
#     "adam_eps": 1e-7,
#     "activation": "gelu",
#     "add_noise": False,
#     "device": "mps",
#     "trg_eq_zero": True,
#     "one_per_run": True,
#     "scale_data": -1,
# }

# all_model_hparams["eternal-sweep-4"] = {
#     "input_dim": 1,
#     "output_dim": 1,
#     "hidden_dim": 4,
#     "num_layers": 6,
#     "num_heads": 4,  # actually 16
#     "pf_dim": 28,
#     "dropout": 0.101129237753374,
#     "seq_length": 101,
#     "lr": 0.0016232282208415774,
#     "weight_decay": 0.00000002093107010601,
#     "adam_eps": 0.00000076632750964993,
#     "activation": "gelu",
#     "add_noise": False,
#     "device": "mps",
#     "trg_eq_zero": True,
#     "one_per_run": True,
#     "scale_data": -1,
#     "max_v": 5.0,
# }

# all_model_hparams["valiant-sweep-5"] = {
#     "activation": "gelu",
#     "adam_eps": 0.00000294831583815948,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.19015934249828412,
#     "hidden_dim": 16,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.00011577143421009044,
#     "max_v": 5.0,
#     "num_heads": 1,
#     "num_layers": 4,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pf_dim": 29,
#     "scale_data": -1,
#     "seq_length": 101,
#     "trg_eq_zero": True,
#     "wandb_project_name": "transformer_anomaly",
#     "weight_decay": 0.00000004232175561521,
# }

# all_model_hparams["lucky-sweep-7"] = {
#     "activation": "gelu",
#     "adam_eps": 0.00000155407451612405,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.11179332021355733,
#     "hidden_dim": 8,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.001013324315796265,
#     "max_v": 5.0,
#     "num_heads": 8,  # actually 16
#     "num_layers": 7,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pf_dim": 46,
#     "scale_data": -1,
#     "seq_length": 101,
#     "trg_eq_zero": True,
#     "wandb_project_name": "transformer_anomaly",
#     "weight_decay": 0.00000004332207635852,
# }

# all_model_hparams["usual-deluge-25"] = {
#     "activation": "gelu",
#     "adam_eps": 0.000001,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.2,
#     "hid_dim": 4,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.0001,
#     "max_v": 5.0,
#     "n_heads": 2,
#     "n_layers": 3,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pf_dim": 48,
#     "scale_data": -1,
#     "seq_length": 101,
#     "trg_eq_zero": True,
#     "wandb_project_name": "new_transformer_anomaly",
#     "weight_decay": 0.00000001,
# }

# all_model_hparams["vocal-butterfly-25"] = {
#     "activation": "gelu",
#     "adam_eps": 0.000001,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.2,
#     "hid_dim": 4,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.0001,
#     "max_lr_factor": 1.5,
#     "max_v": 5.0,
#     "n_heads": 4,
#     "n_layers": 3,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pct_start": 0.1,
#     "pf_dim": 48,
#     "scale_data": -1,
#     "trg_eq_zero": True,
#     "wandb_project_name": "multiloader_transformer_anomaly",
#     "weight_decay": 0.00000001,
#     "pos_enc_scale": 1.0,
#     "sub_cont": False,
#     "scheduler_name": "cycle",
# }

# all_model_hparams["royal-breeze-29"] = {
#     "activation": "gelu",
#     "adam_eps": 0.000001,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.2,
#     "hid_dim": 4,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.0001,
#     "max_lr_factor": 1.5,
#     "max_v": 3.0,
#     "n_heads": 4,
#     "n_layers": 3,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pct_start": 0.1,
#     "pf_dim": 48,
#     "scale_data": -1,
#     "trg_eq_zero": True,
#     "wandb_project_name": "multiloader_transformer_anomaly",
#     "weight_decay": 0.00000001,
#     "sub_cont": True,
#     "pos_enc_scale": 1e-3,
# }

# all_model_hparams["glowing-plasma-34"] = {
#     "activation": "gelu",
#     "adam_eps": 0.000001,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.2,
#     "hid_dim": 4,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.0001,
#     "max_lr_factor": 1.5,
#     "max_v": 3.0,
#     "n_heads": 4,
#     "n_layers": 3,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pct_start": 0.1,
#     "pf_dim": 48,
#     "scale_data": -1.0,
#     "trg_eq_zero": True,
#     "wandb_project_name": "multiloader_transformer_anomaly",
#     "weight_decay": 0.00000001,
#     "sub_cont": True,
#     "pos_enc_scale": 1e-3,
# }

# all_model_hparams["fragrant-sweep-17"] = {
#     "activation": "gelu",
#     "adam_eps": 0.00001849698035835188,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.10808613520471454,
#     "eta_min": 0.00000000217278789093,
#     "gamma": 0.5,
#     "hid_dim": 16,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.0002730096888599789,
#     "max_lr_factor": 1.5,
#     "max_v": 3.0,
#     "n_heads": 4,
#     "n_layers": 5,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pct_start": 0.1,
#     "pf_dim": 63,
#     "pos_enc_scale": 0.9870589760988062,
#     "scale_data": -1.0,
#     "scheduler_name": "cycle",
#     "sub_cont": True,
#     "trg_eq_zero": True,
#     "wandb_project_name": "multiloader_transformer_anomaly",
#     "weight_decay": 0.0000000108237957845,
# }

# all_model_hparams["brisk-sweep-18"] = {
#     "activation": "gelu",
#     "adam_eps": 0.00000118014466862255,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.16944116853569172,
#     "eta_min": 0.00000000047838310321,
#     "gamma": 0.5,
#     "hid_dim": 16,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.000441545716494228,
#     "max_lr_factor": 1.5,
#     "max_v": 3.0,
#     "n_heads": 16,
#     "n_layers": 5,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pct_start": 0.1,
#     "pf_dim": 82,
#     "pos_enc_scale": 0.5341006373829968,
#     "scale_data": -1.0,
#     "scheduler_name": "step",
#     "sub_cont": True,
#     "trg_eq_zero": True,
#     "wandb_project_name": "multiloader_transformer_anomaly",
#     "weight_decay": 0.00000001934702810754
# }

# all_model_hparams["robust-sweep-6"] = {
#     "activation": "gelu",
#     "adam_eps": 0.00002196690928090633,
#     "add_noise": False,
#     "device": "mps",
#     "dropout": 0.16209580941243892,
#     "eta_min": 0.00000000130856243302,
#     "gamma": 0.5,
#     "hid_dim": 16,
#     "ignore_zeros": True,
#     "input_dim": 1,
#     "line_index": 1,
#     "lr": 0.000440375397169194,
#     "max_lr_factor": 1.5,
#     "max_v": 3.0,
#     "n_heads": 1,
#     "n_layers": 5,
#     "one_per_run": True,
#     "output_dim": 1,
#     "pct_start": 0.1,
#     "pf_dim": 57,
#     "pos_enc_scale": 0.36892242917820206,
#     "scale_data": -1.0,
#     "scheduler_name": "step",
#     "sub_cont": True,
#     "trg_eq_zero": True,
#     "wandb_project_name": "multiloader_transformer_anomaly",
#     "weight_decay": 0.0000000140128056027,
# }
# all_model_hparams["chocolate-sweep-33"] = {
#     "activation": "gelu",
#     "adam_eps": 0.00000451806941616253,
#     "add_noise": False,
#     "center": False,
#     "device": "mps",
#     "dropout": 0.14010226950029014,
#     "eta_min": 0.000000001,
#     "gamma": 0.5,
#     "ignore_zeros": True,
#     "input_dim": 80,
#     "latent_dim": 49,
#     "leaky_relu_frac": 0.2,
#     "line_index": 1,
#     "lr": 0.00006499459690895615,
#     "max_lr_factor": 1.5,
#     "max_seq_length": 80,
#     "max_v": 3.0,
#     "mlp_layer_dim": 103,
#     "num_mlp_layers": 8,
#     "one_per_run": True,
#     "output_dim": 80,
#     "pad_value": -100,
#     "pct_start": 0.1,
#     "random_cycle": False,
#     "scale_data": -1.0,
#     "scheduler_name": "cycle",
#     "sub_cont": True,
#     "use_batchnorm": False,
#     "wandb_project_name": "multiloader_autoencoder_anomaly",
#     "weight_decay": 0.00000002061120054049,
#     "weight_init": "xavier",
# }

crops = {
    "MWC_758_12CO_robust0.5_width0.028kms_threshold2.0sigma_taper0.15arcsec.clean.image.fits": (
        (300, 300),
        (300, 300),
    ),
    "Elias27_CO.fits": ((300, 300), (300, 300)),
    "AS209_CO.fits": ((300, 300), (300, 300)),
    "HD142666_CO.fits": ((500, 500), (500, 500)),
    "HD163296_CO.fits": ((150, 150), (150, 150)),
    "HD_97048_13CO.fits": ((200, 200), (200, 200)),
}

channel_limits = {
    "MWC_758_12CO_robust0.5_width0.028kms_threshold2.0sigma_taper0.15arcsec.clean.image.fits": (
        200,
        500,
    ),
    "member.uid___A001_X88a_X2b.Per33_L1448_IRS3B_sci.spw31.cube.I.pbcor.fits": (451, 495),
    "HD_97048_13CO.fits": (25, 155),
}

translations = {
    "MWC_758_12CO_robust0.5_width0.028kms_threshold2.0sigma_taper0.15arcsec.clean.image.fits": "MWC 758",
    "Elias27_CO.fits": "Elias 2-27",
    "HD142666_CO.fits": "HD 142666",
    "HD163296_CO.fits": "HD 163296",
    "HD_97048_13CO.fits": "HD 97048",
    "AS209_CO.fits": "AS 209",
    "member.uid___A001_X88a_X2b.Per33_L1448_IRS3B_sci.spw31.cube.I.pbcor.fits": "L1448 IRS3B",
}

systemic_channels = {
    "MWC_758_12CO_robust0.5_width0.028kms_threshold2.0sigma_taper0.15arcsec.clean.image.fits": 358
    - 200,
    "Elias27_CO.fits": 31,
    "AS209_CO.fits": 25,
    "HD142666_CO.fits": 30,
    "HD163296_CO.fits": 52,
    "HD_97048_13CO.fits": 85 - 25,
    "member.uid___A001_X88a_X2b.Per33_L1448_IRS3B_sci.spw31.cube.I.pbcor.fits": 475 - 451,
}

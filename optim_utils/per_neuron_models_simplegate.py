# ============================================================================
# per_neuron_models.py -- Neural network and physics model classes
# Extracted from per_neuron_trainer_8.py to reduce file size.
# ============================================================================

import os
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from per_neuron_noNN import NONN_MODELS_CONFIG, compute_alpha_eff, compute_vgs_gate
from per_neuron_physics_params import PHYSICS_PARAM_CONFIG

# -----------------------
# Activation Definitions
# -----------------------
class ActivationType:
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SINE = "sin"
    COSINE = "cos"
    SINH = "sinh"
    COSH = "cosh"
    LINEAR = "linear"
    SWISH = "swish"
    MISH = "mish"
    SOFTPLUS = "softplus"

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Cos(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class Sinh(nn.Module):
    def forward(self, x):
        return torch.sinh(x)

class Cosh(nn.Module):
    def forward(self, x):
        return torch.cosh(x)

def get_activation_instance(act_type):
    """Returns a new instance of the activation module."""
    if act_type == ActivationType.RELU: return nn.ReLU()
    elif act_type == ActivationType.TANH: return nn.Tanh()
    elif act_type == ActivationType.SIGMOID: return nn.Sigmoid()
    elif act_type == ActivationType.SINE: return Sin()
    elif act_type == ActivationType.COSINE: return Cos()
    elif act_type == ActivationType.SINH: return Sinh()
    elif act_type == ActivationType.COSH: return Cosh()
    elif act_type == ActivationType.LINEAR: return nn.Identity()
    elif act_type == ActivationType.SWISH: return Swish()
    elif act_type == ActivationType.MISH: return Mish()
    elif act_type == ActivationType.SOFTPLUS: return nn.Softplus()
    else: raise ValueError(f"Unknown activation: {act_type}")


# -----------------------------------------------------------------------------
# Matched weight initialization
# -----------------------------------------------------------------------------
# Per-activation gains = 1 / RMS(activation(z)) for z ~ N(0,1) (forward-variance
# preservation). relu is the exact sqrt(2); the rest were measured empirically
# with that definition. These differ slightly from torch's recommended values
# (e.g. tanh 1.59 here vs 5/3) because torch also folds in backward-gradient
# variance. The gain only scales the INITIAL weights; the forward pass and the
# activation itself are unchanged.
_ACTIVATION_GAINS = {
    "linear":     1.0000,
    "relu":       math.sqrt(2.0),   # 1.4142 (exact)
    "leaky_relu": math.sqrt(2.0),
    "sigmoid":    1.8459,
    "tanh":       1.5924,
    "softplus":   1.0415,
    "swish":      1.6754,
    "mish":       1.4859,
    "sin":        1.5208,
    "cos":        1.3273,
    "sinh":       0.5587,   # <1: heavy-tailed, shrinks weights
    "cosh":       0.4878,
}
_RELU_LIKE = {"relu", "leaky_relu"}

# How to initialize a MIXED-activation layer (homogeneous layers are always
# activation-matched). "xavier": one Xavier(gain=1.0) for the whole matrix.
# "per_activation": each output neuron's row gets its OWN activation's gain
# (Xavier fan), so each neuron's incoming weights respect its activation.
_MIXED_INIT_MODES = ("xavier", "per_activation")
_MIXED_INIT_MODE = "xavier"


def set_mixed_init_mode(mode):
    """Select the init scheme for mixed-activation layers (see _MIXED_INIT_MODE).
    Process-local; call once at startup (each training subprocess sets its own)."""
    global _MIXED_INIT_MODE
    if mode not in _MIXED_INIT_MODES:
        raise ValueError(f"mixed_init mode must be one of {_MIXED_INIT_MODES}, got {mode!r}")
    _MIXED_INIT_MODE = mode
    return _MIXED_INIT_MODE


# vdsgate_aeff family: alpha(Vgs) = transform(polynomial of order N in normalized Vgs),
# lamb(Vgs) = polynomial of order N-1. The wrapper name sets N: lin=1, quad=2, cub=3, quart=4,
# quint=5, sext=6, sept=7 (and the bare "vdsgate_aeff" aliases cub). "_sig" uses the bounded
# 2*sigmoid transform (else softplus); "_clam" forces a constant lamb. E.g. vdsgate_aeff_quart = 4th.
_AEFF_ORDERS = {"vdsgate_aeff": 3, "vdsgate_aeff_lin": 1, "vdsgate_aeff_quad": 2, "vdsgate_aeff_cub": 3,
                "vdsgate_aeff_quart": 4, "vdsgate_aeff_quint": 5, "vdsgate_aeff_sext": 6, "vdsgate_aeff_sept": 7}

def aeff_spec(wrapper):
    """For a vdsgate_aeff* wrapper return (alpha_order, use_sigmoid, lam_order, free_lam); else
    (None, False, None, False). Suffixes compose in any order:
      '_sig'     -> bounded 2*sigmoid alpha (else softplus)
      '_clam'    -> CONSTANT lamb (degree 0, l0 only) instead of the default lamb degree alpha_order-1
      '_freelam' -> unconstrained raw polynomial lambda (old behavior); default is softplus >= 0
    (SIMPLEGATE: the gate is just nn_raw -- whatever output_activation already produced inside
    the network. There is no separate gate-mode setting here; pick softplus/tanh/linear/sigmoid
    via output_activation directly, same as any other wrapper.)"""
    if not wrapper.startswith("vdsgate_aeff"):
        return None, False, None, False
    use_sig  = "_sig" in wrapper
    const_lam = "_clam" in wrapper
    free_lam  = "_freelam" in wrapper
    base = wrapper.replace("_sig", "").replace("_clam", "").replace("_freelam", "")
    order = _AEFF_ORDERS.get(base)
    if order is None:
        return None, use_sig, None, free_lam
    return order, use_sig, (0 if const_lam else order - 1), free_lam


# vdsgate_vdsk family: Vds_knee(Vgs) = softplus(poly of order N) directly models the knee
# voltage in volts. Better conditioned than alpha=1/Vds_knee for low-Vgs curves where the
# physical knee shifts toward 0 V near pinch-off. lamb(Vgs) forced >= 0 via softplus to
# prevent (1+l*Vds) going negative (no Ids<0 and no knee bump). No _sig variant.
_VDSK_ORDERS = {"vdsgate_vdsk": 3, "vdsgate_vdsk_lin": 1, "vdsgate_vdsk_quad": 2, "vdsgate_vdsk_cub": 3}

def vdsk_spec(wrapper):
    """For a vdsgate_vdsk* wrapper return (knee_order, lam_order, free_lam); else (None, None, False).
    '_clam'    -> constant lambda (degree 0).
    '_freelam' -> unconstrained raw polynomial lambda; default is softplus >= 0."""
    if not wrapper.startswith("vdsgate_vdsk"):
        return None, None, False
    const_lam = "_clam" in wrapper
    free_lam  = "_freelam" in wrapper
    base = wrapper.replace("_clam", "").replace("_freelam", "")
    order = _VDSK_ORDERS.get(base)
    if order is None:
        return None, None, free_lam
    return order, (0 if const_lam else order - 1), free_lam


def init_per_neuron_linear(layer, verbose=False):
    """Matched weight init for one PerNeuronLinear.

    - Homogeneous layer (a single activation): that activation's method with its
      fixed gain -> relu-like uses Kaiming/He (std = gain/sqrt(fan_in)), everything
      else uses Xavier/Glorot (std = gain*sqrt(2/(fan_in+fan_out))).
    - Mixed layer (multiple activations): controlled by _MIXED_INIT_MODE:
        * "xavier"         -> one Xavier(gain=1.0) for the whole matrix.
        * "per_activation" -> each output neuron's row scaled by its OWN
                              activation's gain (Xavier fan).

    Init only sets the STARTING weights; nothing in the forward pass changes.
    """
    lin = layer.linear
    fan_in, fan_out = layer.in_features, layer.out_features
    acts = list(layer.activations_list)
    homogeneous = len(set(acts)) == 1
    xavier_base = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        if homogeneous:
            act = acts[0]
            gain = _ACTIVATION_GAINS.get(act, 1.0)
            if act in _RELU_LIKE:
                std = gain / math.sqrt(fan_in)
                method = f"Kaiming({act}, gain={gain:.3f})"
            else:
                std = gain * xavier_base
                method = f"Xavier({act}, gain={gain:.3f})"
            nn.init.normal_(lin.weight, mean=0.0, std=std)
        elif _MIXED_INIT_MODE == "per_activation":
            # Row j (-> neuron j) scaled by neuron j's activation gain. Vectorized:
            # draw N(0,1), then multiply each row by gain_j * xavier_base.
            gains = torch.tensor([_ACTIVATION_GAINS.get(a, 1.0) for a in acts],
                                 dtype=lin.weight.dtype, device=lin.weight.device).unsqueeze(1)
            nn.init.normal_(lin.weight, mean=0.0, std=1.0)
            lin.weight.mul_(gains * xavier_base)
            method = f"PerAct(mixed={sorted(set(acts))})"
        else:  # "xavier"
            nn.init.normal_(lin.weight, mean=0.0, std=xavier_base)  # gain 1.0
            method = f"Xavier(mixed={sorted(set(acts))}, gain=1.0)"
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)
    if verbose:
        print(f"[init] {method}  fan_in={fan_in} fan_out={fan_out}")
    return method

# -----------------------
# Optimized Custom Layer
# # -----------------------
# class PerNeuronLinear(nn.Module):
#     def __init__(self, in_features, out_features, activations_list):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.linear = nn.Linear(in_features, out_features)
        
#         # Save for equation generation
#         self.activations_list = activations_list

#         # Group indices by activation type
#         grouped_indices_temp = {}
#         self.grouped_acts = nn.ModuleDict() 
#         self.act_names = [] # To iterate deterministically

#         for i, act_name in enumerate(activations_list):
#             if act_name not in grouped_indices_temp:
#                 grouped_indices_temp[act_name] = []
#                 self.grouped_acts[act_name] = get_activation_instance(act_name)
#                 self.act_names.append(act_name)
#             grouped_indices_temp[act_name].append(i)

#         # OPTIMIZATION: Register indices as buffers
#         # They will now auto-move to GPU when you call model.to(device)
#         for act_name, indices in grouped_indices_temp.items():
#             # Create tensor
#             idx_tensor = torch.tensor(indices, dtype=torch.long)
            
#             # Sanitize name for safety (e.g. "idx_tanh", "idx_sigmoid")
#             # We use register_buffer so it becomes part of the state_dict
#             self.register_buffer(f"idx_{act_name}", idx_tensor)

#     def forward(self, x):
#         out = self.linear(x)
#         final_out = torch.empty_like(out)
        
#         # Fast Loop: No CPU-GPU transfers here!
#         for act_name in self.act_names:
#             # Retrieve the buffer (already on the correct device)
#             # We use getattr because the attribute name is dynamic
#             indices = getattr(self, f"idx_{act_name}")
            
#             # Select -> Activate -> Copy
#             # Everything happens purely on VRAM
#             subset = torch.index_select(out, 1, indices)
#             activated_subset = self.grouped_acts[act_name](subset)
#             final_out.index_copy_(1, indices, activated_subset)
            
#         return final_out


class PerNeuronLinear(nn.Module):
    def __init__(self, in_features, out_features, activations_list):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        
        # Save for equation generation
        self.activations_list = activations_list

        # Keep a list of unique activation names for the Fast Path check
        self.act_names = list(set(activations_list))
        
        # Store actual activation functions in a ModuleDict
        self.grouped_acts = nn.ModuleDict() 
        for act_name in self.act_names:
            self.grouped_acts[act_name] = get_activation_instance(act_name)

    def forward(self, x):
        out = self.linear(x)
        
        # ðŸŒŸ THE FAST PATH: 99% of layers use a single activation type.
        if len(self.act_names) == 1:
            return self.grouped_acts[self.act_names[0]](out)
            
        # ðŸ›¡ï¸ THE SAFE PATH: For mixed activations. Safe for Double-Backward autograd.
        activated_cols = []
        for i in range(self.out_features):
            act_name = self.activations_list[i]
            col = out[:, i:i+1] # Slice keeping it 2D
            act_col = self.grouped_acts[act_name](col)
            activated_cols.append(act_col)
            
        return torch.cat(activated_cols, dim=1)


# -----------------------
# Dynamic Model
# -----------------------
class DynamicNN(nn.Module):
    def __init__(self, input_dim, neurons_per_layer, activations_per_layer, output_dim, output_activation):
        super().__init__()
        # ---------------------------------------------------------
        # [DYNAMIC NN DEBUGGER - INCOMING ARGUMENTS CHECK]
        # ---------------------------------------------------------
        print(f"\n" + "="*50)
        print(f"[DYNAMIC NN DEBUG] Constructing DynamicNN:")
        print(f"   -> input_dim:             {input_dim}")
        print(f"   -> neurons_per_layer:     {neurons_per_layer}  <-- Is this empty?")
        print(f"   -> activations_per_layer: {activations_per_layer}")
        print(f"   -> output_dim:            {output_dim}")
        print(f"   -> output_activation:     '{output_activation}'  <-- IF THIS IS 'linear', WE CAUGHT THE CULPRIT!")
        print("="*50 + "\n")
        # ---------------------------------------------------------
        
        # CRITICAL FIX: Save these attributes so generate_equation_from_model works
        self.neurons_per_layer = neurons_per_layer
        self.activations_per_layer = activations_per_layer
        
        layers = []
        in_features = input_dim

        # Hidden layers
        for layer_idx, n_out in enumerate(neurons_per_layer):
            activations = activations_per_layer[layer_idx]
            if not activations or len(activations) != n_out:
                raise ValueError(f"Layer {layer_idx}: expected {n_out} activations, got {len(activations)}")
            
            layers.append(PerNeuronLinear(in_features, n_out, activations))
            in_features = n_out

        # Output layer
        # Note: We treat the output layer as a PerNeuronLinear too for consistency
        output_acts = [output_activation] * output_dim
        layers.append(PerNeuronLinear(in_features, output_dim, output_acts))

        self.net = nn.Sequential(*layers)

        # Matched weight init per layer: homogeneous -> activation-matched
        # (Kaiming for relu-like, Xavier otherwise) with the fixed gain; mixed
        # -> Xavier. Replaces PyTorch's default (Kaiming-uniform a=sqrt(5)).
        for _layer in self.net:
            if isinstance(_layer, PerNeuronLinear):
                init_per_neuron_linear(_layer, verbose=True)

    def forward(self, x):
        return self.net(x)
    

# ==========================================
#        PHYSICS PARAMETER CONFIGURATION
# ==========================================

# # Threshold Voltage (Vth)
# VTH_INIT = -2.5
# VTH_MIN  = -3.0
# VTH_MAX  = -2.0

# # Alpha (Knee shape / Saturation)
# ALPHA_INIT = 5.0
# ALPHA_MIN  = 3.0
# ALPHA_MAX  = 8.0

# # Lambda (Output Resistance / Slope)
# # set init to 0.0 to allow it to learn positive or negative direction
# LAMBDA_INIT = 0.0     
# LAMBDA_MIN  = -0.01
# LAMBDA_MAX  = 0.01

# # New Constants for Angelov
# Vpk_INIT, Vpk_MIN, Vpk_MAX = -2.0, -2, -1
# IPK_INIT, IPK_MIN, IPK_MAX = 0.5, 0.3, 1


import torch
import torch.nn as nn
import torch.nn.functional as F




def check_physics_boundaries(top_trials=None, temp_model_dir=None, device=None, physics_c=None, tolerance=0.01, memory_models=None):
    """
    Checks if physical parameters hit the boundaries.
    Can load models from disk (`top_trials` + `temp_model_dir`) OR read them 
    directly from RAM (`memory_models`) to avoid unnecessary saving/loading.
    """
    print("\n" + "="*85)
    print(f"ðŸ” PHYSICAL PARAMETER BOUNDARY CHECK (Tolerance: {tolerance*100}%)")
    print("="*85)

    physics_hits = {}

    # Setup our iterator depending on whether we are reading from RAM or Disk
    if memory_models is not None:
        iterator = [(trial.number, model) for model, trial, *_ in memory_models]
    elif top_trials is not None and temp_model_dir is not None:
        iterator = [(t.number, None) for t in top_trials]
    else:
        print("âš ï¸ No trials or models provided for boundary checking.")
        return physics_hits

    for rank, (t_number, model) in enumerate(iterator, 1):
        trial_hits = []
        
        # --- 1. Load from Disk (Only if memory model wasn't provided) ---
        if model is None:
            trial_filename = os.path.join(temp_model_dir, f"trial_{t_number}.pt")
            if not os.path.exists(trial_filename):
                print(f"âš ï¸ Rank {rank} (Trial {t_number}) - Model not found at {trial_filename}. Skipping.\n")
                continue
                
            try:
                model = torch.load(trial_filename, map_location=device,weights_only=False)
            except Exception as e:
                print(f"âš ï¸ Rank {rank} (Trial {t_number}) - Error loading model: {e}\n")
                continue

        # --- 2. Detect Model Type ---
        mode = getattr(model, 'mode', 'PINN')
        eq_name = getattr(model, 'eq_name', 'Unknown')
        arch_type = 'noNN' if mode == "noNN" else 'PINN'

        print(f"\nðŸ”¹ Rank {rank} (Trial {t_number}) | Arch: {arch_type} ({eq_name}) ðŸ”¹")
        print(f"{'Parameter':<12} | {'Norm (0-1)':<14} | {'Real Value':<14} | {'Boundaries':<18} | {'Status'}")
        print("-" * 85)
        
        warnings_found = False
        
        # --- 3. Gather raw parameters into a dictionary ---
        raw_params_dict = {}
        for name, param in model.named_parameters():
            simple_name = name.split('.')[-1]
            if simple_name.endswith('_raw'):
                base_name = simple_name.replace('_raw', '')
                raw_params_dict[base_name] = param.item() # Storing as floats
                
        if not raw_params_dict:
            continue

        # --- 4. Use Classmethods for Denormalization ---
        try:
            norm_params = PhysicsInformedNN.calc_norm_params(raw_params_dict, mode)
            real_params = PhysicsInformedNN.calc_real_params(raw_params_dict, mode, physics_c, eq_name)
        except KeyError as e:
            print(f"âš ï¸ Configuration error during denormalization: {e}")
            continue

        # --- 5. Evaluate Bounds ---
        for base_name, norm_val in norm_params.items():
            
            # Fetch bounds strictly for terminal display
            try:
                config = PhysicsInformedNN.calc_bound_config(base_name, mode, physics_c, eq_name)
            except KeyError:
                continue 
                
            if not isinstance(config, dict):
                continue
                
            real_val = real_params[base_name]
            raw_val = raw_params_dict[base_name] # Fetch original raw value for the dict payload

            # Evaluate against the squished 0-to-1 normalized value
            status = "âœ… OK"
            if norm_val <= tolerance:
                status = "âš ï¸ HIT LOWER"
                warnings_found = True
                # Maintained the exact keys expected: "param", "limit", "value", "raw_value"
                trial_hits.append({"param": base_name, "limit": "lower", "value": real_val, "raw_value": raw_val})
            elif norm_val >= (1.0 - tolerance):
                status = "âš ï¸ HIT UPPER"
                warnings_found = True
                trial_hits.append({"param": base_name, "limit": "upper", "value": real_val, "raw_value": raw_val})
                
            bounds_str = f"[{config['min']}, {config['max']}]"
            print(f"{base_name:<12} | {norm_val:<14.6e} | {real_val:<14.6e} | {bounds_str:<18} | {status}")

        if trial_hits:
            physics_hits[t_number] = trial_hits

        if not warnings_found:
            print("-" * 85)
            print("âœ… All parameters settled comfortably within their defined ranges.")
        print("\n")

    return physics_hits




class PhysicsInformedNN(nn.Module):

    # SIMPLEGATE variant: there is no separate vdsgate_gate_mode / "tanhm" setting. The
    # vdsgate*/vdsgate_aeff*/vdsgate_vdsk* GATE (the multiplicative magnitude term) is simply
    # nn_raw -- the network's own output, squashed (or not) by whatever output_activation was
    # already set on the network itself. Pick "softplus" for the legacy sign(Ids)=sign(Vds)
    # guarantee, "tanh" for the old "tanhm" behavior (no sign guarantee, better gradient at
    # pinch-off; pair with ids_out_margin), or "linear" for the old vdsgatelin behavior directly.
    # "vdsgatelin" itself is DROPPED (it was a pure alias for "vdsgate" once the gate became
    # output_activation) -- get_eq_details raises a clear error if it's still used in a config.
    # "vdsgate"'s lamb is forced >=0 via softplus by default (prevents Ids<0/bump at high Vds,
    # matching vdsgate_aeff*/vdsgate_vdsk*'s default); "vdsgate_freelam" opts into the old
    # unconstrained (possibly negative) raw lamb.

    def __init__(self, base_nn, equation_type="product", normalization_config=None, config_key=1):
        super().__init__()
        self.base_nn = base_nn
        self.norm_config = normalization_config
        self.config_key = config_key
        # ---------------------------------------------------------
        # [ACTIVATION DEBUGGER - INCOMING NETWORK CHECK]
        # ---------------------------------------------------------
        print(f"\n" + "="*50)
        print(f"[ACTIVATION DEBUG] Inspecting incoming 'base_nn' for equation '{equation_type}'")
        try:
            first_layer = self.base_nn.net[0]
            if hasattr(first_layer, 'activations_list'):
                print(f"   -> [SEARCH] First Layer 'activations_list': {first_layer.activations_list}")
            else:
                print(f"   -> [SEARCH] First Layer doesn't have 'activations_list' attribute.")
            print(f"   -> [SEARCH] First Layer Modules: {first_layer.grouped_acts}")
        except Exception as e:
            print(f"   -> [ERROR] Could not inspect base_nn activations: {e}")
        print("="*50 + "\n")
        # ---------------------------------------------------------
        # CHANGED: Unpack the new aug_dim variable
        self.mode, self.eq_name, self.is_static, self.nn_wrapper, self.aug_dim = self.get_eq_details(equation_type)
        
        print(f"\nPARSER DEBUG: Received '{equation_type}' -> static={self.is_static}, wrapper='{self.nn_wrapper}', mode='{self.mode}', eq_name='{self.eq_name}', aug_dim={self.aug_dim}")

        # NEW: The Auto-Patcher!
        # If the string asked for >2 inputs, but the pipeline gave us a 2-input network, we fix it.
        if self.mode != "noNN" and self.aug_dim > 2:
            first_layer = self.base_nn.net[0]
            if first_layer.in_features != self.aug_dim:
                print(f"AUTO-PATCH: Rewiring first layer of base_nn from {first_layer.in_features} to {self.aug_dim} inputs to support '{equation_type}'!")
                new_layer = PerNeuronLinear(
                    in_features=self.aug_dim,
                    out_features=first_layer.out_features,
                    activations_list=first_layer.activations_list
                )
                init_per_neuron_linear(new_layer)  # keep the matched init on the rewired layer
                self.base_nn.net[0] = new_layer

        if self.mode in ("noNN", "noNN_knee"):
            if self.config_key not in NONN_MODELS_CONFIG:
                raise ValueError(f"Config key {self.config_key} not found in NONN_MODELS_CONFIG dict!")
        else:
            if self.config_key not in PHYSICS_PARAM_CONFIG:
                raise ValueError(f"Config key {self.config_key} not found in PHYSICS_PARAM_CONFIG dict!")

        # --- 0. NORMALIZATION SETUP ---
        self.use_input_norm = False
        self.use_output_norm = False
        self.use_hybrid_norm = False # True = NN is normalized, Physics uses raw real values
        
        self.debug_prints_done = 0
        self.max_debug_prints = 1 

        if self.norm_config:
            self.use_hybrid_norm = self.norm_config.get('hybrid', False)

            # if self.mode in ["noNN"]:
            #     self.use_hybrid_norm = True
            
            if 'vgs_range' in self.norm_config and 'vds_range' in self.norm_config:
                self.use_input_norm = True
                vgs_min, vgs_max = self.norm_config['vgs_range']
                vds_min, vds_max = self.norm_config['vds_range']
                
                self.register_buffer('vgs_min', torch.tensor(vgs_min, dtype=torch.float64))
                self.register_buffer('vgs_scale', torch.tensor(2.0 / (vgs_max - vgs_min), dtype=torch.float64))
                self.register_buffer('vds_min', torch.tensor(vds_min, dtype=torch.float64))
                self.register_buffer('vds_scale', torch.tensor(2.0 / (vds_max - vds_min), dtype=torch.float64))

            if 'ids_range' in self.norm_config:
                self.use_output_norm = True
                ids_min, ids_max = self.norm_config['ids_range']
                self.register_buffer('ids_min', torch.tensor(ids_min, dtype=torch.float64))
                self.register_buffer('ids_scale', torch.tensor(2.0 / (ids_max - ids_min), dtype=torch.float64))

        # --- 3. DEFINE 0-TO-1 OPTIMIZER PARAMETERS ---
        # Shifted default initialization from 0.5 to 0.55 to avoid the exact middle/zero-gradient trap
        def make_param():
            init_val = 0.4 + (torch.rand(1).item() * 0.2)
            return nn.Parameter(torch.tensor(init_val, dtype=torch.float64))
        
        def init_phys_param(p_name):
            # ðŸŒŸ NEW: Skip if we already created this parameter!
            if hasattr(self, f"{p_name}_raw"):
                return 
                
            bound_info = self.get_bound_config(p_name)
            if isinstance(bound_info, dict):
                setattr(self, f"{p_name}_raw", make_param())
            elif isinstance(bound_info, (int, float)):
                fixed_tensor = nn.Parameter(torch.tensor(float(bound_info), dtype=torch.float64), requires_grad=False)
                setattr(self, f"{p_name}_raw", fixed_tensor)

        self.nonn_param_names = []
        self.nonn_static_params = {}

        if self.mode in ("noNN", "noNN_knee"):
            if self.eq_name not in NONN_MODELS_CONFIG[self.config_key]:
                raise ValueError(f"noNN model '{self.eq_name}' not found in NONN_MODELS_CONFIG dict under key {self.config_key}!")
                
            func = NONN_MODELS_CONFIG[self.config_key][self.eq_name]['func']
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())[1:] # Skip 'X'
            
            for p_name in param_names:
                bound_info = self.get_bound_config(p_name)
                # Check if it's a static number instead of a config dictionary
                if isinstance(bound_info, (int, float)):
                    self.nonn_static_params[p_name] = float(bound_info)
                else:
                    setattr(self, f"{p_name}_raw", make_param())
                    self.nonn_param_names.append(p_name)


        elif not self.is_static:
            if self.nn_wrapper in ("vdsgate", "vdsgate_freelam"):
                # New structured pure-NN drain equation:
                #   Ids = gate(NN(Vgs,Vds)) * tanh(alpha*Vds) * (1 + lamb*Vds)
                # tanh(alpha*Vds) supplies the exact zero at Vds=0 and the odd reverse branch
                # (alpha forced > 0 via softplus in forward). gate = nn_raw, i.e. whatever
                # output_activation already produced (softplus -> sign(Ids)=sign(Vds)
                # guaranteed; linear -> more flexible, no sign guarantee).
                # lamb forced >=0 via softplus by default (prevents Ids<0/bump at high Vds,
                # same convention as vdsgate_aeff*/vdsgate_vdsk*'s default); "vdsgate_freelam"
                # opts into the old unconstrained (possibly negative) raw lamb.
                # Self-contained learnable params (independent of the physics bound-config).
                self.vdsgate_alpha_raw = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
                self.vdsgate_lamb_raw  = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            elif self.nn_wrapper.startswith("vdsgate_aeff") or self.nn_wrapper.startswith("vdsgate_vdsk"):
                # vdsgate with a Vgs-DEPENDENT knee (two sub-families sharing the same param layout):
                #   aeff: alpha(Vgs)=transform(poly N), lamb(Vgs)=softplus(poly M) >= 0
                #         Ids = softplus(NN) * tanh(alpha(Vgs)*Vds) * (1 + lamb(Vgs)*Vds)
                #   vdsk: Vds_knee(Vgs)=softplus(poly N)>0, lamb(Vgs)=softplus(poly M) >= 0
                #         Ids = softplus(NN) * tanh(Vds/Vds_knee(Vgs)) * (1 + lamb(Vgs)*Vds)
                # Name sets N (lin=1, quad=2, cub=3; bare ==cub); "_clam" -> M=0 else M=N-1.
                if self.nn_wrapper.startswith("vdsgate_aeff"):
                    _ord, _, _lord, _ = aeff_spec(self.nn_wrapper)
                else:
                    _ord, _lord, _ = vdsk_spec(self.nn_wrapper)
                if _ord is None:
                    raise ValueError(f"unknown vdsgate variant: {self.nn_wrapper!r}")
                for _i in range(_ord + 1):                    # knee/alpha coeffs a0..a_N
                    setattr(self, f"vdsgate_a{_i}",
                            nn.Parameter(torch.tensor(0.5 if _i == 0 else 0.0, dtype=torch.float64)))
                for _j in range(_lord + 1):                   # lamb coeffs l0..l_M (M = _lord)
                    setattr(self, f"vdsgate_l{_j}", nn.Parameter(torch.tensor(0.0, dtype=torch.float64)))
            if not (self.mode == "pure" and (self.nn_wrapper in ("identity", "vdsgate", "vdsgate_freelam")
                                             or self.nn_wrapper.startswith("vdsgate_aeff")
                                             or self.nn_wrapper.startswith("vdsgate_vdsk"))):
                if self.nn_wrapper != "angelovwrap":
                    init_phys_param("vth")
                    init_phys_param("alpha")
                    init_phys_param("lamb")
            if self.nn_wrapper == "angelovwrap":
                init_phys_param("Ipk")
                init_phys_param("alpha")
                init_phys_param("lamb")
            
            if self.eq_name.startswith("angelov"):
                init_phys_param("Vpk")
                init_phys_param("Ipk")

                if "quad" in self.eq_name or "cubic" in self.eq_name:
                    init_phys_param("P1")
                    init_phys_param("P2")
                if "cubic" in self.eq_name:
                    init_phys_param("P3")

            elif self.eq_name in ["square_law", "self_heating", "trap_dip", "crossover_slope"]:
                init_phys_param("P1")
                if self.eq_name in ["self_heating", "crossover_slope"]: 
                    init_phys_param("delta")
                if self.eq_name == "trap_dip":
                    init_phys_param("dip_amp")
                    init_phys_param("dip_center") 
                    init_phys_param("dip_width")
                    
            elif self.eq_name == "sigmoid": 
                init_phys_param("scale")
            elif self.eq_name == "double_tanh": 
                init_phys_param("k_gate")
        
        # --- 4. INITIALIZE NN OUTPUT ---
        if self.mode != "noNN":
            final_layer = self.base_nn.net[-1].linear
            
            if self.mode == "pure":
                # Standard initialization for pure NN to learn quickly
                nn.init.xavier_uniform_(final_layer.weight, gain=1.0)
                nn.init.constant_(final_layer.bias, 0.0)
            else:
                # Zero initialization for PINNs so physics takes the lead initially
                nn.init.constant_(final_layer.weight, 0.0) 
                nn.init.constant_(final_layer.bias, 0.0)

        # --- 5. INITIALIZATION DEBUGGER ---
        print(f"\nDEBUG: Verifying Initial Parameter Mapping for '{self.eq_name}'...")
        initial_real_params = self.get_real_params()
        for p_name, real_val in initial_real_params.items():
            if hasattr(self, f"{p_name}_raw"):
                raw_val = getattr(self, f"{p_name}_raw").item()
                raw_print = raw_val.item() if hasattr(raw_val, 'item') else float(raw_val)
                real_print = real_val.item() if hasattr(real_val, 'item') else float(real_val)
                print(f"   -> {p_name:<12}: Raw = {raw_print:<6.4f} | Real = {real_print:+.6e}")
                if real_val == 0.0:
                    print(f"      ðŸš¨ CRITICAL WARNING: '{p_name}' evaluates to exactly 0.0! Gradients will likely be zero.")
        print("-" * 75)
 
 
     # --- PARAMETER MANAGEMENT ---
    
    
    def get_bound_config(self, name):
        """ Wrapper for the class method. """
        return self.calc_bound_config(name, self.mode, self.config_key, getattr(self, 'eq_name', None))

    def get_real_params(self):
        """ Wrapper that gathers instance tensors and passes them to the class method. """
        keys_to_check = self.nonn_param_names if self.mode in ("noNN", "noNN_knee") else PHYSICS_PARAM_CONFIG[self.config_key].keys()
        
        # Build a dictionary of the raw tensors currently on the model
        raw_dict = {}
        for name in keys_to_check:
            if hasattr(self, f"{name}_raw"):
                raw_dict[name] = getattr(self, f"{name}_raw")
                
        return self.calc_real_params(
            raw_params_dict=raw_dict, 
            mode=self.mode, 
            config_key=self.config_key, 
            eq_name=getattr(self, 'eq_name', None), 
            nonn_static_params=getattr(self, 'nonn_static_params', None)
        )

    def get_norm_params(self):
        """ Wrapper that gathers instance tensors and passes them to the class method. """
        keys_to_check = self.nonn_param_names if self.mode in ("noNN", "noNN_knee") else PHYSICS_PARAM_CONFIG[self.config_key].keys()
        
        raw_dict = {}
        for name in keys_to_check:
            if hasattr(self, f"{name}_raw"):
                raw_dict[name] = getattr(self, f"{name}_raw")
                
        return self.calc_norm_params(
            raw_params_dict=raw_dict, 
            mode=self.mode, 
            nonn_static_params=getattr(self, 'nonn_static_params', None)
        )

    def get_active_params(self):
        """ Returns real bounds for pure physics or hybrid mode, else standard normalized. """
        # ðŸŒŸ FIX: Pure physics (noNN) MUST always use real physical bounds
        if self.mode == "noNN":
            return self.get_real_params()
            
        if self.use_input_norm and not self.use_hybrid_norm:
            return self.get_norm_params()
        else:
            return self.get_real_params()

    # --- PHYSICS FORWARD EQUATIONS ---
    def physical_forward(self, Vgs, Vds):
        """ Computes I_physics using chosen scaling. """
        if self.is_static: return torch.tanh(5.0 * Vds) * (1.0 + 0.001 * Vds)

        p = self.get_active_params()

        if self.mode in ("noNN", "noNN_knee"):
            # 1. Retrieve the target function from your config dictionary
            func = NONN_MODELS_CONFIG[self.config_key][self.eq_name]['func']
            
            # 2. Reconstruct the 'X' tensor [batch_size, 2] that the function expects
            X = torch.cat([Vgs, Vds], dim=1)
            
            # 3. Gather all parameters (both learnable and static) into a kwargs dictionary
            func_kwargs = {}
            for p_name in self.nonn_param_names:
                # Grabs the activated/scaled parameter from your p dictionary
                func_kwargs[p_name] = p[p_name] 
                
            for p_name, static_val in getattr(self, 'nonn_static_params', {}).items():
                func_kwargs[p_name] = static_val
                
            # 4. Execute the mathematical function and return immediately
            out = func(X, **func_kwargs)
            if out.dim() == 1:
                out = out.unsqueeze(1) # Force it back to [batch_size, 1]
            return out

        # --- ANGELOV OPTIONS ---
        if self.eq_name == "angelov_drain":
            return p["Ipk"] * torch.tanh(p["alpha"] * Vds) * (1 + p["lamb"] * Vds)
        elif self.eq_name == "angelov_drain_nolam":
            return p["Ipk"] * torch.tanh(p["alpha"] * Vds)
        elif self.eq_name == "angelov_quad":
            x = Vgs - p["Vpk"]
            psi = (p['P1'] * x) + (p['P2'] * (x**2))
            gate = p["Ipk"] * (1 + torch.tanh(psi))
            return gate * torch.tanh(p["alpha"] * Vds) * (1 + p["lamb"] * Vds)
        elif self.eq_name == "angelov_cubic":
            x = Vgs - p["Vpk"]
            psi = (p['P1'] * x) + (p['P2'] * (x**2)) + (p['P3'] * (x**3))
            gate = p["Ipk"] * (1 + torch.tanh(psi))
            return gate * torch.tanh(p["alpha"] * Vds) * (1 + p["lamb"] * Vds)

        # --- STANDARD OPTIONS ---
        elif self.eq_name == "basic":
            return torch.tanh(p["alpha"] * Vds) * (1 + p["lamb"] * Vds)
        elif self.eq_name == "basic_positive":
            return torch.tanh(p["alpha"] * Vds) * (1 + torch.abs(p["lamb"]) * Vds)
        elif self.eq_name == "square_law":
            x = Vgs - p["vth"]
            Vov = 0.5 * x * (1 + torch.tanh(p['P1'] * x))
            return (Vov ** 2) * torch.tanh(p["alpha"] * Vds)
        elif self.eq_name == "sigmoid":
            diff = (Vgs - p["vth"]) * p["scale"]
            sig = torch.sigmoid(diff)
            return ((Vgs - p["vth"]) * sig) ** 2 * torch.tanh(p["alpha"] * Vds)
        elif self.eq_name == "double_tanh":
            gate = 0.5 * (1 + torch.tanh(p["k_gate"] * (Vgs - p["vth"])))
            return gate * torch.tanh(p["alpha"] * Vds)
        elif self.eq_name == "self_heating":
            x = Vgs - p["vth"]
            Vov = 0.5 * x * (1 + torch.tanh(p['P1'] * x))
            I_base = (Vov ** 2) * torch.tanh(p["alpha"] * Vds)
            return I_base * torch.exp(-p["delta"] * Vds)
        elif self.eq_name == "trap_dip":
            x = Vgs - p["vth"]
            Vov = 0.5 * x * (1 + torch.tanh(p['P1'] * x))
            I_base = (Vov ** 2) * torch.tanh(p["alpha"] * Vds)
            gauss = torch.exp(-((Vds - p["dip_center"])**2) / (2 * p["dip_width"]**2))
            return torch.relu(I_base - p["dip_amp"] * gauss)
        elif self.eq_name == "crossover_slope":
            x = Vgs - p["vth"]
            Vov = 0.5 * x * (1 + torch.tanh(p['P1'] * x))
            I_base = (Vov ** 2) * torch.tanh(p["alpha"] * Vds)
            slope_factor = p["lamb"] - (p["delta"] * Vov)
            return I_base * (1.0 + slope_factor * Vds)
        
        return torch.zeros_like(Vds)


    def forward(self, *args):
        # --- 0. INPUT UNPACKING ---
        if len(args) == 1:
            # Optuna/Training Loop Format: model(X) where X is [batch_size, 2]
            X = args[0]
            Vgs = X[:, 0].unsqueeze(1)
            Vds = X[:, 1].unsqueeze(1)
        elif len(args) == 2:
            # Explicit Format: model(Vgs, Vds)
            Vgs, Vds = args[0], args[1]
            if Vgs.dim() == 1: Vgs = Vgs.unsqueeze(1)
            if Vds.dim() == 1: Vds = Vds.unsqueeze(1)
        else:
            raise TypeError(f"forward() expected 1 or 2 positional arguments, got {len(args)}")

        # --- 1. ROUTING & NORMALIZATION ---
        if self.use_input_norm:
            if getattr(self, 'use_hybrid_norm', False):
                # Physics gets RAW. NN gets NORMALIZED.
                phys_vgs, phys_vds = Vgs, Vds
                nn_vgs = (Vgs - self.vgs_min) * self.vgs_scale - 1.0
                nn_vds = (Vds - self.vds_min) * self.vds_scale - 1.0
            else:
                # Fallback: Inputs are already normalized. Denorm for physics.
                nn_vgs, nn_vds = Vgs, Vds
                phys_vgs = (Vgs + 1.0) / self.vgs_scale + self.vgs_min
                phys_vds = (Vds + 1.0) / self.vds_scale + self.vds_min
        else:
            phys_vgs, phys_vds = Vgs, Vds
            nn_vgs, nn_vds = Vgs, Vds

        # --- 2. PHYSICS BASELINE ---
        phys_out = self.physical_forward(phys_vgs, phys_vds)

        if self.mode == "noNN":
            final_out = phys_out
        elif self.mode == "noNN_knee":
            # Gate: g = 1 - tanh(alpha_gate/k * Vds)
            # knee_use_alpha_eff=True  → alpha_gate = alpha_eff(Vgs,Vds) from physics (per-sample)
            # knee_use_alpha_eff=False → alpha_gate = static alphaR/alpha from frozen params
            # k = knee_alpha_scale widens the gate into early saturation (default 1.0)
            # knee_combiner: 'sum'     → phys + gate*NN
            #                'product' → phys * (1 + gate*NN)
            real_p = self.get_real_params()
            k = float(getattr(self, 'knee_alpha_scale', 1.0))
            if getattr(self, 'knee_use_alpha_eff', False):
                alpha_gate = compute_alpha_eff(self.eq_name, phys_vgs, phys_vds, real_p)
            else:
                alpha_gate = real_p.get('alpha', real_p.get('alphaR', None))
            if alpha_gate is not None:
                # alpha_gate may be a tensor (trainable / mod1 expression) or a
                # plain float (frozen scalar from nonn_static_params, e.g. classic_angelov
                # with freeze_physics=True). Promote to tensor so torch.abs() works.
                if not torch.is_tensor(alpha_gate):
                    alpha_gate = torch.as_tensor(
                        float(alpha_gate), dtype=phys_vds.dtype, device=phys_vds.device
                    )
                gate = (1.0 - torch.tanh(torch.abs(alpha_gate) / k * phys_vds)).detach()
            else:
                gate = torch.ones_like(phys_vds)
            nn_input = torch.cat([nn_vgs, nn_vds], dim=1)
            nn_out = self.base_nn(nn_input)
            
            # DUMP GATE INFO TO CONSOLE
            if getattr(self, 'debug_prints_done_gate', 0) == 0:
                self.debug_prints_done_gate = 1
            combiner = getattr(self, 'knee_combiner', 'sum')
            if combiner == 'product':
                # gated multiplicative: phys * (1 + gate * NN)
                # NN output is a unitless fraction; gate zeros it in deep saturation
                final_out = phys_out * (1.0 + gate * nn_out)
            elif combiner == 'sum_gated_vgs':
                # gated additive with extra Vgs conduction gate h(Vgs,Vds):
                # phys + g(Vds) * h(Vgs) * NN
                # h → 0 at pinch-off, h = 1 at peak gm, h → 2 deep on
                h = compute_vgs_gate(self.eq_name, phys_vgs, phys_vds, real_p).detach()
                if self.use_output_norm and self.use_hybrid_norm:
                    nn_out = nn_out / self.ids_scale
                final_out = phys_out + gate * h * nn_out
            elif combiner == 'residual':
                # bounded multiplicative: phys * (1 + alpha_max * g_Vds * g_Vgs * tanh(NN))
                #
                # g_Vds (knee gate, computed above): suppresses NN in deep saturation.
                # g_Vgs (optional sigmoid Vgs gate): suppresses NN below a Vgs threshold,
                #   fixing the Gm bump at low Vgs.
                #   g_Vgs = sigmoid((Vgs - vgs_thr) / vgs_tau)
                #   At Vgs << vgs_thr : g_Vgs → 0 → NN inactive (physics dominates).
                #   At Vgs >> vgs_thr : g_Vgs → 1 → NN fully active.
                #   vgs_thr : threshold voltage [physical V].  Default: None (gate disabled).
                #   vgs_tau : softness [V, >0].  Default: 0.3.
                #
                # knee_max_correction ∈ (0,1] further limits NN amplitude.
                _alpha_max = float(getattr(self, 'knee_max_correction', 1.0))
                _vgs_thr   = getattr(self, 'knee_vgs_thr', None)
                _vgs_tau   = float(getattr(self, 'knee_vgs_tau', 0.3))
                if _vgs_thr is not None:
                    _vgs_tau = max(_vgs_tau, 1e-3)   # avoid div-by-zero
                    g_vgs = torch.sigmoid((phys_vgs - float(_vgs_thr)) / _vgs_tau)
                    final_out = phys_out * (1.0 + _alpha_max * gate * g_vgs * torch.tanh(nn_out))
                else:
                    final_out = phys_out * (1.0 + _alpha_max * gate * torch.tanh(nn_out))
            else:  # 'sum' (default)
                # gated additive: phys + gate * NN
                if self.use_output_norm and self.use_hybrid_norm:
                    nn_out = nn_out / self.ids_scale
                final_out = phys_out + gate * nn_out
        else:
            # --- 3. APPLY FEATURE AUGMENTATION ---
            # Driven dynamically by self.aug_dim extracted during initialization
            if getattr(self, 'aug_dim', 2) == 2:
                # Standard Mode (Backwards Compatible)
                nn_input = torch.cat([nn_vgs, nn_vds], dim=1)
                
            elif self.aug_dim == 4:
                # Augmented Mode (Physics Context Injected)
                p = self.get_active_params()
                
                # Calculate Effective Overdrive Voltage safely
                v_th = p.get("Vpk", p.get("vth", torch.zeros_like(phys_vgs)))
                v_drive = phys_vgs - v_th
                
                # Normalize the physical current so it matches the [-1, 1] NN domain
                if self.use_output_norm:
                    phys_feat = (phys_out.detach() - self.ids_min) * self.ids_scale - 1.0
                else:
                    phys_feat = phys_out.detach() 
                    
                # Stack all 4 features
                nn_input = torch.cat([nn_vgs, nn_vds, phys_feat, v_drive], dim=1)
                
            else:
                raise ValueError(f"Unsupported aug_dim={self.aug_dim}. Expected 2 or 4.")

            # --- 4. NEURAL NETWORK FORWARD ---
            nn_raw = self.base_nn(nn_input)
            
            # Apply Wrapper
            if self.nn_wrapper == "tanh": nn_act = torch.tanh(nn_raw)
            elif self.nn_wrapper == "sigmoid": nn_act = torch.sigmoid(nn_raw)
            elif self.nn_wrapper == "softplus": nn_act = F.softplus(nn_raw)
            elif self.nn_wrapper == "exp": nn_act = torch.exp(nn_raw)
            elif self.nn_wrapper == "invexp": nn_act = 1.0 - torch.exp(-torch.abs(nn_raw))
            elif self.nn_wrapper == "angelovwrap":
                p = self.get_active_params() # Grab the raw parameters
                nn_act = p["Ipk"] * (1 + torch.tanh(nn_raw)) * torch.tanh(p["alpha"] * phys_vds) * (1 + p["lamb"] * phys_vds)
            elif self.nn_wrapper in ("vdsgate", "vdsgate_freelam"):
                # Ids = gate(NN) * tanh(alpha*Vds) * (1 + lamb*Vds).
                # tanh(alpha*Vds) -> exact 0 at Vds=0 and odd reverse branch; alpha=softplus(raw)>0.
                # SIMPLEGATE: gate = nn_raw, i.e. whatever output_activation already produced --
                # no separate gate-mode setting. Pick the gate shape (softplus/tanh/linear/sigmoid)
                # via output_activation itself.
                # lamb: softplus(raw)>=0 by default (no Ids<0/bump); "vdsgate_freelam" -> raw,
                # unconstrained (can go negative -- e.g. to fit self-heating current droop).
                g_alpha = F.softplus(self.vdsgate_alpha_raw)
                g_gate  = nn_raw
                l_eff = (self.vdsgate_lamb_raw if self.nn_wrapper == "vdsgate_freelam"
                         else F.softplus(self.vdsgate_lamb_raw))
                nn_act = g_gate * torch.tanh(g_alpha * phys_vds) * (1.0 + l_eff * phys_vds)
            elif self.nn_wrapper.startswith("vdsgate_aeff"):
                # Vgs-dependent knee. alpha(Vgs) = transform(poly N); lamb(Vgs) = poly M.
                # Default: l_eff = softplus(poly) >= 0 (prevents Ids<0 / bump).
                # _freelam suffix: l_eff = raw poly (unconstrained, old behavior).
                # SIMPLEGATE: gate = nn_raw (output_activation IS the gate; no separate setting).
                _ord, _use_sig, _lord, _free_lam = aeff_spec(self.nn_wrapper)
                vg = nn_vgs
                poly_a = sum(getattr(self, f"vdsgate_a{_i}") * vg**_i for _i in range(_ord + 1))
                poly_l = sum(getattr(self, f"vdsgate_l{_j}") * vg**_j for _j in range(_lord + 1))
                a_eff  = (2.0 * torch.sigmoid(poly_a)) if _use_sig else F.softplus(poly_a)
                l_eff  = poly_l if _free_lam else F.softplus(poly_l)
                nn_act = nn_raw * torch.tanh(a_eff * phys_vds) * (1.0 + l_eff * phys_vds)
            elif self.nn_wrapper.startswith("vdsgate_vdsk"):
                # Vgs-dependent knee VOLTAGE. Vds_knee(Vgs) = softplus(poly N) > 0 models the knee
                # position in volts — better gradient signal for low-Vgs curves where the knee shifts
                # toward 0 V near pinch-off.
                # Default: l_eff = softplus(poly) >= 0 (no Ids<0). _freelam: raw poly (unconstrained).
                # SIMPLEGATE: gate = nn_raw (output_activation IS the gate; no separate setting).
                _ord, _lord, _free_lam = vdsk_spec(self.nn_wrapper)
                vg = nn_vgs
                poly_k = sum(getattr(self, f"vdsgate_a{_i}") * vg**_i for _i in range(_ord + 1))
                poly_l = sum(getattr(self, f"vdsgate_l{_j}") * vg**_j for _j in range(_lord + 1))
                Vds_knee = F.softplus(poly_k) + 1e-4   # > 0, small floor avoids div-by-zero
                l_eff    = poly_l if _free_lam else F.softplus(poly_l)
                nn_act = nn_raw * torch.tanh(phys_vds / Vds_knee) * (1.0 + l_eff * phys_vds)
            else: nn_act = nn_raw  # identity

            # --- 5. COMBINER ---
            if self.mode == "sum":
                if self.use_hybrid_norm and self.use_output_norm:
                    # Convert NN output from normalized domain to physical Amps
                    nn_act = nn_act / self.ids_scale 
                combined = phys_out + nn_act
                
            elif self.mode == "product":
                # Math: Real Amps * (1.0 + unitless percentage) = Real Amps
                combined = phys_out * (1.0 + nn_act)
                
            elif self.mode == "physics":
                combined = phys_out
                
            else: # pure
                combined = nn_act


            # --- COMBINER DEBUGGER (Relative Contributions) ---
            if getattr(self, 'debug_prints_done', 0) < getattr(self, 'max_debug_prints', 1):
                # Only run this once to avoid destroying your console
                with torch.no_grad():
                    phys_mag = torch.mean(torch.abs(phys_out)).item()
                    nn_mag = torch.mean(torch.abs(nn_act)).item()
                    
                    if self.mode == "sum":
                        total_mag = phys_mag + nn_mag + 1e-9
                        print(f"\nðŸ¥Š TUG-OF-WAR DEBUG (SUM MODE):")
                        print(f"   -> Avg Physics Contribution: {phys_mag:.4f} A ({(phys_mag/total_mag)*100:.1f}%)")
                        print(f"   -> Avg NN Contribution:      {nn_mag:.4f} A ({(nn_mag/total_mag)*100:.1f}%)")
                    
                    elif self.mode == "product":
                        # In product, nn_act is a percentage scaler
                        print(f"\nðŸ¥Š TUG-OF-WAR DEBUG (PRODUCT MODE):")
                        print(f"   -> Avg Physics Base: {phys_mag:.4f} A")
                        print(f"   -> Avg NN Scaler:    {nn_mag*100:.2f} % adjustment")
                        
                self.debug_prints_done += 1

            final_out = combined

        # --- 6. FINAL DENORMALIZATION ---
        # If hybrid is true, combined is already in real Amps, so skip this.
        if self.use_output_norm and not getattr(self, 'use_hybrid_norm', False) and self.mode not in ["noNN", "noNN_knee"]:
            final_out = (final_out + 1.0) / self.ids_scale + self.ids_min

        return final_out


    @classmethod
    def get_eq_details(cls, equation_type):
        """
        Parses strings formatted as: mode:wrapper-eq_name-augX
        Example: "product:tanh-angelov_cubic-aug4"
              modes: noNN | physics: $I_{pred} = I_{angelov}$
                    pure: $I_{pred} = I_{nn}$
                    sum: $I_{pred} = I_{angelov} + I_{nn}$
                    product: $I_{pred} = I_{angelov} * (1 + I_{nn})$
        """
        equation_type = equation_type.lower()
        
        # --- 1. Fallbacks for simple static modes ---
        if equation_type in ["product", "sum"]:
            return equation_type, "static_fallback", True, "identity", 2
        if equation_type == "pure":
            return "pure", "none", False, "identity", 2

        # --- 2. Extract Mode (Everything before the ':') ---
        if ":" in equation_type:
            mode, rest = equation_type.split(":", 1)
        else:
            mode = "product"  # Safe default
            rest = equation_type
            
        # Clean up mode naming (e.g., nonn -> noNN)
        if mode == "nonn": 
            mode = "noNN"
        elif mode == "nonn_knee":
            mode = "noNN_knee"

        # --- 3. Extract Components (Separated by '-') ---
        parts = rest.split("-")
        
        aug_dim = 2 
        nn_wrapper = "identity"
        eq_name = "static_fallback"
        is_static = False

        # Check if the last part is the augmentation dimension (e.g., 'aug4')
        if parts[-1].startswith("aug"):
            try:
                aug_dim = int(parts[-1].replace("aug", ""))
                parts.pop(-1) # Remove it from the list
            except ValueError:
                pass

        # "vdsgatelin" was DROPPED (SIMPLEGATE): it behaved identically to "vdsgate" once the
        # gate became output_activation directly. Fail loudly instead of silently falling
        # through to nn_wrapper="identity" (which would silently disable the wrapper entirely).
        if parts and parts[0] == "vdsgatelin":
            raise ValueError(
                "wrapper 'vdsgatelin' was removed in the SIMPLEGATE model -- it is now "
                "identical to 'vdsgate' (the gate is output_activation itself). Use "
                "'vdsgate' instead; pick the old vdsgatelin behavior via "
                "output_activation='linear'."
            )

        # Check if the first part is our wrapper (e.g., 'tanh')
        # vdsgate_aeff* covers the whole knee family (lin/quad/cub x optional _sig) via startswith.
        valid_wrappers = ["tanh", "sigmoid", "softplus", "exp", "invexp", "angelovwrap", "vdsgate", "vdsgate_freelam", "vdsgate_aeff", "vdsgate_aeff_lin", "vdsgate_aeff_quad", "vdsgate_aeff_cub", "vdsgate_aeff_lin_sig", "vdsgate_aeff_quad_sig", "vdsgate_aeff_cub_sig", "vdsgate_aeff_sig", "vdsgate_vdsk", "vdsgate_vdsk_lin", "vdsgate_vdsk_quad", "vdsgate_vdsk_cub", "identity"]
        if parts and (parts[0] in valid_wrappers or parts[0].startswith("vdsgate_aeff") or parts[0].startswith("vdsgate_vdsk")):
            nn_wrapper = parts.pop(0)

        # Whatever is left in the middle is the equation name!
        if parts:
            eq_name = "-".join(parts) # Re-join in case the equation name strangely has a dash

        return mode, eq_name, is_static, nn_wrapper, aug_dim

    @classmethod
    def calc_bound_config(cls, name, mode, config_key, eq_name=None):
        """ Checks bounds independent of a model instance. """
        if mode in ("noNN", "noNN_knee"):
            if eq_name in NONN_MODELS_CONFIG.get(config_key, {}):
                local_bounds = NONN_MODELS_CONFIG[config_key][eq_name].get('bounds', {})
                if name in local_bounds: 
                    return local_bounds[name]
            # STRICT ENFORCEMENT
            raise KeyError(f"Parameter '{name}' is missing from the 'bounds' dict of noNN model '{eq_name}' under key {config_key}!")
            
        # Fallback for PINN modes only
        return PHYSICS_PARAM_CONFIG.get(config_key, {}).get(name, {"min": -10.0, "max": 10.0, "lr": 0.1})

    @classmethod
    def calc_real_params(cls, raw_params_dict, mode, config_key, eq_name=None, nonn_static_params=None):
        """ Maps 0-to-1 raw parameters into their real-world physical bounds. """
        real_params = {}
        
        for name, raw_val in raw_params_dict.items():
            config = cls.calc_bound_config(name, mode, config_key, eq_name)
            
            # 1. Is it a learnable parameter? (Needs scaling)
            if isinstance(config, dict):
                # Handle PyTorch tensors (during training)
                if isinstance(raw_val, torch.Tensor):
                    p_norm = torch.sigmoid(raw_val) # Keeps the gradient graph!
                # Handle standard floats (from YAML/Pickle extraction script)
                else:
                    p_norm = 1.0 / (1.0 + math.exp(-max(min(float(raw_val), 60), -60)))
                    
                real_val = config["min"] + p_norm * (config["max"] - config["min"])
                real_params[name] = real_val
                
            # 2. Is it a fixed constant? (Pass it straight through)
            elif isinstance(config, (int, float)):
                # raw_val is ALREADY the tensor/float we created, so just assign it directly
                real_params[name] = raw_val 
                
        if mode in ("noNN", "noNN_knee") and nonn_static_params:
            real_params.update(nonn_static_params)
            
        return real_params

    @classmethod
    def calc_norm_params(cls, raw_params_dict, mode, nonn_static_params=None):
        """ Returns the 0-to-1 normalized parameters directly. """
        norm_params = {}
        
        for name, raw_val in raw_params_dict.items():
            if isinstance(raw_val, torch.Tensor):
                norm_params[name] = torch.sigmoid(raw_val)
            else:
                norm_params[name] = 1.0 / (1.0 + math.exp(-max(min(float(raw_val), 60), -60)))
                
        if mode == "noNN" and nonn_static_params:
            norm_params.update(nonn_static_params)
            
        return norm_params


    def get_physics_str(self):
        def fmt(val): return f"{val:.6e}"
        p = {k: v if isinstance(v, (int, float)) else v.item() for k, v in self.get_active_params().items()}

        # 1. INPUT ROUTING (Restored to your original, correct logic)
        if self.mode != "noNN" and self.use_input_norm and not getattr(self, 'use_hybrid_norm', False):
            # hybrid=False: Physics block expects normalized inputs. 
            # We normalize the raw simulator inputs (_v1, _v2).
            v1_str = f"((_v1 - {fmt(self.vgs_min)}) * {fmt(self.vgs_scale)} - 1.0)"
            v2_str = f"((_v2 - {fmt(self.vds_min)}) * {fmt(self.vds_scale)} - 1.0)"
        else:
            # hybrid=True: Physics block expects raw inputs. Pass directly.
            v1_str, v2_str = "_v1", "_v2"

        nn_core = "NN"

        if self.nn_wrapper == "identity":   nn_expr = nn_core
        elif self.nn_wrapper == "tanh":     nn_expr = f"tanh({nn_core})"
        elif self.nn_wrapper == "sigmoid":  nn_expr = f"(1.0/(1.0+exp(-{nn_core})))"
        elif self.nn_wrapper == "softplus": nn_expr = f"ln(1.0+exp({nn_core}))"
        elif self.nn_wrapper == "exp":      nn_expr = f"exp({nn_core})"
        elif self.nn_wrapper == "invexp":   nn_expr = f"exp(-{nn_core})"
        elif self.nn_wrapper == "angelovwrap":
            a, l = p["alpha"], p["lamb"]
            base_expr = f"((1.0+tanh({nn_core})) * tanh({fmt(a)}*{v2_str}) * (1.0+{fmt(l)}*{v2_str}))"
            
            if self.mode in ["pure", "sum"]:
                ipk = p["Ipk"]
                nn_expr = f"({fmt(ipk)} * {base_expr})"
            else:
                nn_expr = base_expr
        else: nn_expr = nn_core

        eq_str = "0.0"
        
        # ... [Physics Equation Block remains exactly the same] ...
        if self.mode == "noNN":
            eq_str = NONN_MODELS_CONFIG[self.config_key][self.eq_name]["str_template"](v1_str,v2_str,fmt,p)

        elif self.eq_name == "static_fallback": eq_str = f"(tanh(5.0*{v2_str})*(1.0+0.001*{v2_str}))"
        elif self.eq_name == "angelov_drain": eq_str = f"({fmt(p['Ipk'])} * tanh({fmt(p['alpha'])}*{v2_str}) * (1.0+{fmt(p['lamb'])}*{v2_str}))"
        elif self.eq_name == "angelov_drain_nolam": eq_str = f"({fmt(p['Ipk'])} * tanh({fmt(p['alpha'])}*{v2_str}))"
        elif self.eq_name == "angelov_quad":
            x = f"({v1_str}-({fmt(p['Vpk'])}))"
            psi = f"({fmt(p['P1'])}*{x} + {fmt(p['P2'])}*{x}**2)"
            gate = f"({fmt(p['Ipk'])} * (1.0 + tanh({psi})))"
            eq_str = f"({gate} * tanh({fmt(p['alpha'])}*{v2_str}) * (1.0 + {fmt(p['lamb'])}*{v2_str}))"
        elif self.eq_name == "angelov_cubic":
            x = f"({v1_str}-({fmt(p['Vpk'])}))"
            psi = f"({fmt(p['P1'])}*{x} + {fmt(p['P2'])}*{x}**2 + {fmt(p['P3'])}*{x}**3)"
            gate = f"({fmt(p['Ipk'])} * (1.0 + tanh({psi})))"
            eq_str = f"({gate} * tanh({fmt(p['alpha'])}*{v2_str}) * (1.0 + {fmt(p['lamb'])}*{v2_str}))"
        elif self.eq_name == "basic": eq_str = f"(tanh({fmt(p['alpha'])}*{v2_str})*(1.0+{fmt(p['lamb'])}*{v2_str}))"
        elif self.eq_name == "basic_positive": eq_str = f"(tanh({fmt(p['alpha'])}*{v2_str})*(1.0+{abs(p['lamb']):.3e}*{v2_str}))"
        elif self.eq_name == "square_law":
            x = f"({v1_str}-({fmt(p['vth'])}))"
            vov = f"(0.5*{x}*(1.0+tanh({fmt(p['P1'])}*{x})))"
            eq_str = f"({vov}**2 * tanh({fmt(p['alpha'])}*{v2_str}))"
        elif self.eq_name == "sigmoid":
            diff = f"(({v1_str}-({fmt(p['vth'])})) * {fmt(p['scale'])})"
            sig = f"(1.0 / (1.0 + exp(-{diff})))"
            eq_str = f"((({v1_str}-({fmt(p['vth'])})) * {sig})**2 * tanh({fmt(p['alpha'])}*{v2_str}))"
        elif self.eq_name == "double_tanh":
            eq_str = f"((0.5*(1.0+tanh({fmt(p['k_gate'])}*({v1_str}-({fmt(p['vth'])}))))) * tanh({fmt(p['alpha'])}*{v2_str}))"
        elif self.eq_name == "self_heating":
            x = f"({v1_str}-({fmt(p['vth'])}))"
            vov = f"(0.5*{x}*(1.0+tanh({fmt(p['P1'])}*{x})))"
            eq_str = f"({vov}**2 * tanh({fmt(p['alpha'])}*{v2_str}) * exp(-{fmt(p['delta'])}*{v2_str}))"
        elif self.eq_name == "trap_dip":
            x = f"({v1_str}-({fmt(p['vth'])}))"
            vov = f"(0.5*{x}*(1.0+tanh({fmt(p['P1'])}*{x})))"
            base = f"({vov}**2 * tanh({fmt(p['alpha'])}*{v2_str}))"
            dip = f"({fmt(p['dip_amp'])} * exp(-(({v2_str}-({fmt(p['dip_center'])}))**2)/(2.0*{fmt(p['dip_width'])}**2)))"
            eq_str = f"max(0, {base} - {dip})"
        elif self.eq_name == "crossover_slope":
             x = f"({v1_str}-({fmt(p['vth'])}))"
             vov = f"(0.5*{x}*(1.0+tanh({fmt(p['P1'])}*{x})))"
             base = f"({vov}**2 * tanh({fmt(p['alpha'])}*{v2_str}))"
             slope = f"(1.0 + ({fmt(p['lamb'])} - ({fmt(p['delta'])}*{vov})) * {v2_str})"
             eq_str = f"({base} * {slope})"

        # 2. OUTPUT COMBINATION
        if self.mode == "product":
            if self.nn_wrapper == "angelovwrap": combined = f"({eq_str} * {nn_expr})"
            else: combined = f"({eq_str} * (1.0 + {nn_expr}))"
        elif self.mode == "sum": 
            # FIX: If hybrid, physics is in raw Amps but NN is normalized. 
            # We must scale the NN magnitude before adding it to the physics baseline.
            if getattr(self, 'use_hybrid_norm', False) and self.use_output_norm:
                nn_expr_scaled = f"({nn_expr} / {fmt(self.ids_scale)})" # Note: Check if your forward pass uses / or * here
                combined = f"({eq_str} + {nn_expr_scaled})"
            else:
                combined = f"({eq_str} + {nn_expr})"
        elif self.mode == "pure":
            combined = nn_expr
        elif self.mode.startswith("noNN"):
            combined = eq_str
        else: combined = "ERROR_MODE"

        # 3. FINAL DENORMALIZATION
        # FIX: Removed the `pure` mode exception. If hybrid=False, the ENTIRE combined output 
        # is normalized and needs to be shifted and scaled back to real Amps.
        if self.use_output_norm and not getattr(self, 'use_hybrid_norm', False) and not self.mode.startswith("noNN"): 
            return f"(({combined} + 1.0) / {fmt(self.ids_scale)} + {fmt(self.ids_min)})"
        
        return combined

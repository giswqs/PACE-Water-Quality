import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

class SparseDispatcher(object):
 
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # sort
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class VAE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # encoder
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_mu = nn.Linear(64, 64) 
        self.fc_logvar = nn.Linear(64, 64)  

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, output_dim))
    def encode(self, x):
        x = self.encoder_layer(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var
     
class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, noisy_gating=False, k=4):
        super(MoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        #self.hidden_size = hidden_size
        self.k = k
        self.experts = nn.ModuleList([VAE(self.input_size, self.output_size) for i in range(self.num_experts)])

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        # TODO
        
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        #top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load
    
    def forward(self, x, loss_coef=1e-2):
        raw_gates, load = self.noisy_top_k_gating(x, self.training)
        
        importance = raw_gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, raw_gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        y = dispatcher.combine([output[0] for output in expert_outputs])  
        
        mu = torch.cat([output[1] for output in expert_outputs], dim=0)
        log_var = torch.cat([output[2] for output in expert_outputs], dim=0)
        return y, mu, log_var, loss
    
class TokenMoEModel(nn.Module):
    def __init__(self, input_size, output_size, num_experts, k=4, token_size=16, stride=8):
        super(TokenMoEModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.k = k
        self.token_size = token_size
        self.stride = stride

        # === unfold相关参数 ===
        self.padding_needed = (stride - (input_size - token_size) % stride) % stride
        self.padded_size = input_size + self.padding_needed
        self.num_tokens = 1 + (self.padded_size - token_size) // stride

        self.unfold = nn.Unfold(kernel_size=(1, token_size), stride=(1, stride))

        # === Token级别MoE结构 ===
        self.moe = MoE(token_size, output_size, num_experts, k=k)

        # === Token输出拼接后的维度 ===
        self.moe_output_dim = output_size * self.num_tokens

        # === 融合VAE结构 ===
        self.fc_mu = nn.Linear(self.moe_output_dim, 64)
        self.fc_log_var = nn.Linear(self.moe_output_dim, 64)

        self.decoder_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_size)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.shape[0]

        # padding
        if self.padding_needed > 0:
            pad = torch.zeros(batch_size, self.padding_needed, device=x.device)
            x = torch.cat([x, pad], dim=1)

        # unfold to tokens
        x = x.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, input_size]
        tokens = self.unfold(x).transpose(1, 2)  # [B, num_tokens, token_size]
        tokens = tokens.reshape(-1, self.token_size)  # [B*num_tokens, token_size]

        # MoE token-wise 处理
        moe_output, mu_token, logvar_token, moe_loss = self.moe(tokens)
        token_outputs = moe_output.reshape(batch_size, -1)  # [B, output_size * num_tokens]

        # 融合后进入 VAE bottleneck
        mu = self.fc_mu(token_outputs)
        log_var = self.fc_log_var(token_outputs)
        z = self.reparameterize(mu, log_var)

        pred_y = self.decoder_layers(z)

        return {
            "pred_y": pred_y,
            "mu": mu,
            "log_var": log_var,
            "moe_loss": moe_loss
        }
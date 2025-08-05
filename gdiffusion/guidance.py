import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_cond_fn(log_prob_fn, latent_dim: int, guidance_strength: float = 1.0, clip_grad=False, clip_grad_max=10.0, debug=False):
    '''
        log_prob_fn --> maps a latent z of shape (B, 128) into a log probability
        guidance_strength --> the guidance strength of the model
        latent_dim --> the latent dim (always 128)
        clip_grad --> if the model should clip the gradient to +-clip_grad_max

        Returns a cond_fn that evaluastes the grad of the log probability
    '''

    def cond_fn(mean, t, **kwargs):
        # mean.shape = (B, 1, 128), so reshape to (B, 128) so predicter can handle it
        mean = mean.detach().reshape(-1, latent_dim)
        mean.requires_grad_(True)

        # if debug:
            # print(f"mean: {mean}")
            # print(f"mean.shape: {mean.shape}")
            # print(f"mean.requires_grad: {mean.requires_grad}")


        #---------------------------------------------------------------------------

        with torch.enable_grad():
            predicted_log_probability = log_prob_fn(mean)
            if debug:
                print(f"pred_log_prob: {predicted_log_probability}")
                print(f"pred_log_prob.shape: {predicted_log_probability.shape}")
                print(f"pred_log_prob.requires_grad {predicted_log_probability.requires_grad}")
                
            gradients = torch.autograd.grad(predicted_log_probability, mean, retain_graph=True)[0]

            # if debug:
                # print(f"gradients: {gradients}")
                # print(f"graidents.shape: {gradients.shape}")
                # print(f"gradients.requires_grad {gradients.requires_grad}")
                
            if clip_grad:
                if debug:
                    print(f"Clipping gradients to {-clip_grad_max} to {clip_grad_max}")
                gradients = torch.clamp(gradients, -clip_grad_max, clip_grad_max)
                
            grads = guidance_strength * gradients.reshape(-1, 1, latent_dim)
            if debug:
                # print(f"grads: {grads}")
                print(f"grad_norm: {grads.norm(2)}")
                print(f"grads.shape: {grads.shape}")
                # print(f"grads.requires_grad {grads.requires_grad}")
                
            return grads
        
    return cond_fn

def get_cond_fn_normal_analytical(mean=0.0, sigma=0.001):
    # analytically computes the gradient of the log probability under
    # a normal distribution with mean=mean, sigma=sigma

    def cond_fn(z, t, **guidance_kwargs):
        z = z.to(device)
        grad = -(z - mean) / (sigma**2)

        # MUST clamp gradient for numerical stability
        # It is REALLY finnicky about how much you clamp
        # not ideal really...

        grad = torch.clamp(grad, min=-100.0, max=100.0)
        return grad
    
    return cond_fn

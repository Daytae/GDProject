from scipy import stats

def is_different_from_other(z, z_other, alpha=0.05, do_print=False):
    guided_flat = z.flatten().detach().cpu().numpy()
    baseline_flat = z_other.flatten().detach().cpu().numpy()
    
    # KS test
    _, p_value = stats.ks_2samp(guided_flat, baseline_flat)
    is_different = p_value < alpha
    
    if do_print:
        print(f"Sample of shape: {z.shape} is {'' if is_different else 'not '}different from other with p={p_value:.4f}\n")
    
    return p_value < alpha, p_value

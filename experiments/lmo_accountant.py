




from check_privacy_mnist import *



def get_complete_privacy(epoch):
    try:
        jsonpath=sys.argv[1]
        lmo = json.loads(Path(jsonpath).read_text())
        lmo['distributions'] = DEFAULT_DISTRIBUTIONS
        lmo['delta'] = DEFAULT_DELTA
    except:
        lmo={
            # "a1": 0.1, "a3": 0.1, "a4": 0.1,
            # "G_theta": 0.5, "G_k": 1, "E_lambda": 5, "U_b": 2, "U_a": 1,
            # "distributions": DEFAULT_DISTRIBUTIONS,
            # "delta": DEFAULT_DELTA,

            # "a1": 0.2, "a3": 0.2, "a4": 0.3,
            # "G_theta": 1.0, "G_k": 2.0, "E_lambda": 5, "U_b": 1, "U_a": 0,
            # "distributions": DEFAULT_DISTRIBUTIONS,
            # "delta": DEFAULT_DELTA,

            "a1": 0.1, "a3": 0.4, "a4": 0.1,
            "G_theta": 5.0, "G_k": 1.0, "E_lambda": 0.5, "U_b": 1, "U_a": 0,
            "distributions": DEFAULT_DISTRIBUTIONS,
            "delta": DEFAULT_DELTA,

        }
    overall_epsilon_lmo, opt_order, rdp_lmo = compute_privacy_lmo(lmo)

    overall_epsilon, sigma = compute_overall_privacy(overall_epsilon_lmo, lmo['delta'], dataset="MNIST")

    overall_epsilon['eps_rdp']  = (overall_epsilon['eps_rdp'] *epoch)

    return overall_epsilon, sigma 





from check_privacy_mnist import *



def get_complete_privacy(epoch, params , dataset, steps=938):
    # try:
    #     jsonpath=sys.argv[1]
    #     lmo = json.loads(Path(jsonpath).read_text())
    #     lmo['distributions'] = DEFAULT_DISTRIBUTIONS
    #     lmo['delta'] = DEFAULT_DELTA
    # except:
    #     lmo={
    #         # "a1": 0.1, "a3": 0.1, "a4": 0.1,
    #         # "G_theta": 0.5, "G_k": 1, "E_lambda": 5, "U_b": 2, "U_a": 1,
    #         # "distributions": DEFAULT_DISTRIBUTIONS,
    #         # "delta": DEFAULT_DELTA,

    #         # "a1": 0.2, "a3": 0.2, "a4": 0.3,
    #         # "G_theta": 1.0, "G_k": 2.0, "E_lambda": 5, "U_b": 1, "U_a": 0,
    #         # "distributions": DEFAULT_DISTRIBUTIONS,
    #         # "delta": DEFAULT_DELTA,

    #         #0.5
    #         # "a1": 0.3,
    #         # "a3": 0.1,
    #         # "a4": 0.5,
    #         # "G_theta": 1.0,
    #         # "G_k": 2.0,
    #         # "E_lambda": 5,
    #         # "U_b": 1,
    #         # "U_a": 0,
    #         # "distributions": DEFAULT_DISTRIBUTIONS,
    #         # "delta": DEFAULT_DELTA,

    #         # eps =1.5

    #         "a1": 0.9,
    #         "a3": 0.1,
    #         "a4": 0.5,
    #         "G_theta": 1.0,
    #         "G_k": 2.0,
    #         "E_lambda": 1,
    #         "U_b": 2,
    #         "U_a": 1,
    #          "distributions": DEFAULT_DISTRIBUTIONS,
    #         "delta": DEFAULT_DELTA,

    #         #eps =0.8
    #         # "a1": 0.9,
    #         # "a3": 0.2,
    #         # "a4": 0.8,
    #         # "G_theta": 1.0,
    #         # "G_k": 2.0,
    #         # "E_lambda": 5,
    #         # "U_b": 1,
    #         # "U_a": 0,
    #         #  "distributions": DEFAULT_DISTRIBUTIONS,
    #         # "delta": DEFAULT_DELTA,


    #     }
    overall_epsilon_lmo, opt_order, rdp_lmo = compute_privacy_lmo(lmo=params, steps=epoch)

    # adding effect of T ... 
    overall_epsilon_lmo = overall_epsilon_lmo * np.sqrt(epoch)

    overall_epsilon, sigma = compute_overall_privacy(overall_epsilon_lmo, params['delta'], dataset=dataset, steps=steps)

    return overall_epsilon_lmo, sigma 


def get_log_epsilon(epsilon, gamma):
    """
    Calculate the logarithmic expression given gamma and x.

    Parameters:
        gamma (float): The parameter gamma in the equation.
        x (float): The exponent variable in the equation.

    Returns:
        float: The result of the logarithmic expression.
    """
    # Calculate the inner expression
    inner_expression = 1 + gamma * (np.exp(epsilon) - 1)

    # Calculate the logarithm of the inner expression
    result = np.log(inner_expression)

    return result

# if __name__=="__main__":
#    lmo = {            "a1": 0.9,
#             "a3": 0.2,
#             "a4": 0.8,
#             "G_theta": 1.0,
#             "G_k": 2.0,
#             "E_lambda": 5,
#             "U_b": 1,
#             "U_a": 0,
#              "distributions": DEFAULT_DISTRIBUTIONS,
#             "delta": DEFAULT_DELTA,}
#    for i in range(200):
   
#         x, y, z=  compute_privacy_lmo(lmo, i)


#         log_x = get_log_epsilon(x, 64/60000 )

#         print(f"x = {x} and log x {log_x} \n")

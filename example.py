from email import iterators
import numpy as np
import pandas as pd

from black_litterman_entropy_pooling import (
    probability_constraint, 
    rank_view, 
    entropy_program, 
    merge_prior_posterior
)


if __name__ == "__main__":
    # fictional ER (expected returns) and Vol values
    data = [
        {"asset_class": "Credit - High Yield",   "ER": 0.08, "Vol": 0.15},
        {"asset_class": "Equity - US Small",     "ER": 0.06, "Vol": 0.25},
        {"asset_class": "Bond - INT Treasuries", "ER": 0.04, "Vol": 0.10},
        {"asset_class": "Credit - REITs",        "ER": 0.02, "Vol": 0.06},
        {"asset_class": "Alternative - Gold",    "ER": 0.00, "Vol": 0.12},
    ]
    df = pd.DataFrame(data).set_index("asset_class")
    er = df['ER']
    vol = df['Vol']

    # let's make up time series values to generate a covariance matrix from it
    assets = list(df.index.values)
    np.random.seed(0)
    cov = pd.DataFrame(np.random.randn(800, len(assets)), columns=assets).cov()

    mu = er
    sigma = cov

    j = 1000000
    # iterations = j
    
    # generate a bunch of "random" expected returns based  mean Expected Return and the asset covariance matrix
    x = np.random.multivariate_normal(mu, sigma, j)
    # er_random_samples = x 
    p = np.ones([j, 1]) / j

    ranks = [
        # 'Alternative - Gold',
        'Credit - High Yield', 
        'Equity - US Small', 
        'Bond - INT Treasuries', 
        'Credit - REITs', 
        'Alternative - Gold'
    ]
    rank_index = [er.index.get_loc(r) for r in ranks]

    Aeq, beq = probability_constraint(x)
    A, b = rank_view(x, p, rank_index[0:-1], rank_index[1:])

    p_ = entropy_program(p, A, b, Aeq, beq)

    # this seems confusing
    # let's find a more direct way to get to, [mu_, sigma_]
    ps = {}
    ps['Prior'] = pd.Series(dict(zip(range(0, len(p)), p.flatten())))
    ps['Posterior'] = pd.Series(dict(zip(range(0, len(p)), p_.flatten())))
    ps = pd.DataFrame(ps)
    mu_, sigma_ = merge_prior_posterior(p, p_, x, 1.)
    mu_ = pd.Series(mu_.flatten(), index=er.index)

    expected_returns = {}
    expected_returns['Prior'] = er
    expected_returns['Posterior'] = mu_

    erdf = pd.DataFrame(expected_returns)
    print(erdf)

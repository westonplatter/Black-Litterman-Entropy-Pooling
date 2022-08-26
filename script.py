import numpy
import scipy.optimize
import pandas

def probability_constraint(x):
    j, n = x.shape

    Aeq = numpy.ones([1, j])
    beq = numpy.array([1.])

    return [Aeq, beq]


def rank_view(x, p, lower, upper):
    j, n = x.shape
    k = len(lower)

    v = x[:, lower] - x[:, upper]
    A = v.transpose()

    if A.ndim == 1:
        A = A.reshape(1, j)

    b = numpy.zeros([A.shape[0], 1])

    return [A, b]


def mean_qualitative_view(x, p, c, multiplier):
    """
    given a panel x
    and probabilities p
    for column c
    m is a multiplier against the vol (e.g. -2, -1, 0, 1, 2, etc)
    """

    j, n = x.shape

    m = numpy.mean(x[:, c])
    s = numpy.std(x[:, c])

    A = x[:, c].transpose().reshape(1, j)
    b = numpy.array([m + multiplier * s])

    return [A, b]


def mean_qualitative_relative_view(x, p, first, second, multiplier):
    """
    given a panel x
    and probabilities p
    for the difference between first and second
    m is a multiplier against the vol (e.g. -2, -1, 0, 1, 2, etc)
    """

    j, n = x.shape

    v = x[:, first] - x[:, second]
    m = numpy.mean(v)
    s = numpy.std(v)

    A = v.transpose().reshape(1, j)
    b = numpy.array([m + multiplier * s])

    return [A, b]


def median_view(x, p, c, q):
    """
    given a panel x
    and probabilities p
    for column c
    set view that the median will be greater than quintile q
    """

    j, n = x.shape

    v = numpy.abs(x[:, c])
    i_sort = numpy.argsort(v)
    v_sort = v[i_sort]

    f = numpy.cumsum(p[i_sort])

    i_ref = numpy.max(numpy.where(f <= q))
    v_ref = v_sort[i_ref]

    i_select = numpy.where(v <= v_ref)

    a = numpy.zeros(1, j)
    a[i_select] = 1.

    A = a
    b = numpy.array([0.5])

    return [A, b]


def volatility_qualitative_view(x, p, c, multiplier):
    """
    multiplier is between (0, infinity)
    """
    j, n = x.shape

    m = numpy.mean(x[:, c])
    s = numpy.std(x[:, c])

    A = numpy.square(x[:, c] - m).transpose().reshape(1, j)
    b = numpy.array([m ** 2 + (multiplier * s) ** 2])

    return [A, b]


def correlation_view(x, p, first, second, corr):
    """
    given a panel x
    and probabilities p
    set view that the correlations between first and second will be c
    """

    j, n = x.shape
    v = x[:, first] * x[:, second]

    m = numpy.mean(x[:, first]) * numpy.mean(x[:, second])
    s = numpy.std(x[:, first]) * numpy.std(x[:, second])

    Aeq = v.transpose().reshape(1, j)
    beq = numpy.array([m + s * corr])
    
    return [Aeq, beq]


def entropy_program(p, A, b, Aeq, beq):
    """
    p - The set of prior probabilities (1 x j)
    A - matrix of inequality constraints (paired with b) (k_ x n)
    b - vector consisting of inequality constraints (1 x k_)
    Aeq - matrix of equality constraints (paired with beq) (k x n)
    beq - vector consisting of equality constraints (1 x k)
    """

    k_ = A.shape[0] # the number of inequality constraints
    k = Aeq.shape[0] # the number of equality constraints

    if k_ + k < 0:
        raise Exception("Must have at least 1 equality or inequality view")

    if abs(numpy.sum(p) - 1.) > 1e-8:
        raise Exception("Probabilities must sum to 1.")

    if Aeq.shape[0] != beq.shape[0]:
        raise Exception("Rows in Aeq must equal rows in beq")

    if A.shape[0] != b.shape[0]:
        raise Exception("Rows in A must equal rows in b")

    if p.shape[1] != 1:
        raise Exception("p must be jx1 shaped")

    A_ = A.transpose()
    b_ = b.transpose()

    Aeq_ = Aeq.transpose()
    beq_ = beq.transpose()

    x0 = numpy.zeros([k_ + k, 1]) # starting guess for optimization; length = number of views 

    # if we only have equality constraints
    if k_ == 0:

        def gradient_u(v):
            v = v.reshape(len(v), 1)
            x = numpy.exp(numpy.log(p) - 1 - Aeq_.dot(v))
            x = numpy.clip(x, 1e-32, numpy.inf)
            return beq - Aeq.dot(x)

        """
        def hessian_u(v):
            v = v.reshape(len(v), 1)
            x = numpy.exp(numpy.log(p) - 1 - Aeq_.dot(v))
            x = numpy.clip(x, 1e-32, numpy.inf)
            return Aeq.dot(numpy.multiply(x.dot(numpy.ones(1, k)), Aeq_)) # Hessian computed by Chen Qing, Lin Daimin, Meng Yanyan, Wang
        """

        def fmin_u(v):
            v = v.reshape(len(v), 1)
            x = numpy.exp(numpy.log(p) - 1 - Aeq_.dot(v))
            x = numpy.clip(x, 1e-32, numpy.inf)
            L = x.transpose().dot(numpy.log(x) - numpy.log(p) + Aeq_.dot(v)) - beq_.dot(v)

            return -L

        result = scipy.optimize.minimize(fmin_u, x0, method = 'L-BFGS-B', jac = gradient_u, tol=1e-6, options = {'ftol': 1e2 * numpy.finfo(float).eps})
        
        if not result.success:
            raise Exception("Optimization failed.  Status " + str(result.status) + ".  Cause: " + result.message)

        #print result

        v = result.x
        v = v.reshape(len(v), 1)
        p_ = numpy.exp(numpy.log(p) - 1 - Aeq_.dot(v))


    # inequality constraints are specified
    else: 
        inq_mat = -numpy.eye(k_ + k)
        inq_mat = inq_mat[:k_,:]

        inq_constraint = lambda x: inq_mat.dot(x)
        jac_constraint = lambda x: inq_mat
    
        def gradient_c(lv):
            lv = lv.reshape(len(lv), 1)
            
            l = lv[:k_]
            v = lv[k_:]
            x = numpy.exp(numpy.log(p) - 1 - A_.dot(l) - Aeq_.dot(v))
            x = numpy.clip(x, 1e-32, numpy.inf)

            return numpy.vstack((b - A.dot(x), beq - Aeq.dot(x)))

        def fmin_c(lv):
            lv = lv.reshape(len(lv), 1)

            log_p = numpy.log(p)

            l = lv[:k_]
            v = lv[k_:]
            x = numpy.exp(log_p - 1 - A_.dot(l) - Aeq_.dot(v))
            x = numpy.clip(x, 1e-32, numpy.inf)

            ineq = A.dot(x) - b
            eq = Aeq.dot(x) - beq

            L = x.transpose().dot(numpy.log(x) - log_p) + l.transpose().dot(ineq) + v.transpose().dot(eq)

            return -L

        cons = {'type': 'ineq', 
                'fun': inq_constraint, 
                'jac': jac_constraint}

        result = scipy.optimize.minimize(fmin_c, x0, method='SLSQP', jac = gradient_c, constraints = cons, tol=1e-6, options = {'ftol': 1e2 * numpy.finfo(float).eps})
        
        if not result.success:
            raise Exception("Optimization failed.  Status " + str(result.status) + ".  Cause: " + result.message)

        #print result 

        lv = result.x
        lv = lv.reshape(len(lv), 1)
        l = lv[0:k_]
        v = lv[k_:]
        p_ = numpy.exp(numpy.log(p) - 1 - A_.dot(l) - Aeq_.dot(v))


    if not (abs(1. - numpy.sum(p_)) < 1e-3):
        print(numpy.sum(p_))
        raise Exception("Sum of posterior probabilities is not equal to 1.")

    return p_


def merge_prior_posterior(p, p_, x, c):
    if (c < -1e8) or (c > (1 + 1e-8)):
        raise Exception("Confidence must be in [0, 1]")

    j, n = x.shape

    p_ = (1. - c) * p + c * p_
    exps = x.transpose().dot(p_)

    scnd_mom = x.transpose().dot(numpy.multiply(x, p_.dot(numpy.ones([1, n]))))
    scnd_mom = (scnd_mom + scnd_mom.transpose()) / 2

    covs = scnd_mom - exps.dot(exps.transpose())

    return exps, covs


"""
if __name__ == "__main__":

    er = pandas.read_csv("cmas/er-vol.csv", index_col = 0, parse_dates = False)['ER'] / 100.
    vol = pandas.read_csv("cmas/er-vol.csv", index_col = 0, parse_dates = False)['Vol'] / 100.
    corr = pandas.read_csv("cmas/correlations.csv", index_col = 0, parse_dates = False)

    covariance = numpy.diag(vol.values).dot(corr).dot(numpy.diag(vol.values))
    covariance = pandas.DataFrame(covariance, index = corr.index, columns = corr.columns)

    mu = er
    sigma = covariance

    j = 1000000

    x = numpy.random.multivariate_normal(mu, sigma, j)
    p = numpy.ones([j, 1]) / j

    ranks = ['Credit - High Yield', 'Equity - US Small', 'Bond - INT Treasuries', 'Credit - REITs', 'Alternative - Gold']
    rank_index = [er.index.get_loc(r) for r in ranks]

    Aeq, beq = probability_constraint(x)
    A, b = rank_view(x, p, rank_index[0:-1], rank_index[1:])

    p_ = entropy_program(p, A, b, Aeq, beq)

    ps = {}
    ps['Prior'] = pandas.Series(dict(zip(range(0, len(p)), p.flatten())))
    ps['Posterior'] = pandas.Series(dict(zip(range(0, len(p)), p_.flatten())))

    ps = pandas.DataFrame(ps)
    ps.to_csv("ps.csv")

    mu_, sigma_ = merge_prior_posterior(p, p_, x, 1.)

    mu_ = pandas.Series(mu_.flatten(), index = er.index)

    expected_returns = {}
    expected_returns['Prior'] = er
    expected_returns['Posterior'] = mu_

    pandas.DataFrame(expected_returns).to_csv("expected-returns.csv")
"""
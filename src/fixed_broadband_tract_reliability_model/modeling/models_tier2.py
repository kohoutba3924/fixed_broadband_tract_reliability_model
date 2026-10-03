from pygam import LinearGAM, s


def build_gam(n_features: int):
    """
    Build a LinearGAM with one spline term per feature.
    """
    terms = sum([s(i) for i in range(n_features)], start=s(0))
    return LinearGAM(terms=terms, fit_intercept=True)

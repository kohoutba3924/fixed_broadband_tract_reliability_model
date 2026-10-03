from sklearn.linear_model import ElasticNet, Ridge


def build_ridge(alpha=1.0):
    """
    Build a Ridge regression model.
    """
    return Ridge(alpha=alpha, random_state=42)


def build_elasticnet(alpha=1.0, l1_ratio=0.5):
    """
    Build an ElasticNet regression model.
    """
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

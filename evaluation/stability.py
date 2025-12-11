import torch


def evaluate_stability(search_fn, model, x, n_runs=5):
    """
    Runs the boundary search multiple times; measures consistency.

    Returns:
        {
            'success_rate': float,
            'mean_pairwise_distance': float,
            'std_to_mean': float
        }
    """
    model.eval()
    boundary_points = []

    for _ in range(n_runs):
        b, success = search_fn(model, x)
        if success:
            boundary_points.append(b.view(-1).detach().cpu())

    successes = len(boundary_points)
    if successes <= 1:
        return {
            "success_rate": successes / n_runs,
            "mean_pairwise_distance": float("nan"),
            "std_to_mean": float("nan")
        }

    stack = torch.stack(boundary_points)
    mean_point = stack.mean(dim=0)

    # pairwise distances
    dists = []
    for i in range(successes):
        for j in range(i + 1, successes):
            d = torch.norm(stack[i] - stack[j], p=2).item()
            dists.append(d)

    mean_pairwise = sum(dists) / len(dists)

    std_to_mean = torch.norm(stack - mean_point.unsqueeze(0), dim=1).std().item()

    return {
        "success_rate": successes / n_runs,
        "mean_pairwise_distance": mean_pairwise,
        "std_to_mean": std_to_mean,
    }

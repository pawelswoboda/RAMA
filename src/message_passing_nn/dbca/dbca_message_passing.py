import torch

class ClassicalMessagePassing:
    def __init__(self, edge_costs, corr_12, corr_13, corr_23, t12, t13, t23, edge_counter):
        
        self.edge_costs = edge_costs
        self.corr_12 = corr_12
        self.corr_13 = corr_13
        self.corr_23 = corr_23
        self.t12 = t12
        self.t13 = t13
        self.t23 = t23
        self.edge_counter = edge_counter

        assert all(t.device == edge_costs.device for t in [
            corr_12, corr_13, corr_23,
            t12, t13, t23, edge_counter
        ])

    def compute_edge_lower_bound(self):
        return torch.sum(torch.where(self.edge_costs < 0, self.edge_costs, torch.zeros_like(self.edge_costs)))

    def compute_triangle_lower_bound(self):
        a = self.t12
        b = self.t13
        c = self.t23
        zero = torch.zeros_like(a)

        lb = torch.stack([
            zero,
            a + b,
            a + c,
            b + c,
            a + b + c
        ])
        return torch.min(lb, dim=0).values.sum()

    def compute_lower_bound(self):
        return self.compute_edge_lower_bound() + self.compute_triangle_lower_bound()

    def send_messages_to_triplets(self):
        edge_vals_12 = self.edge_costs[self.corr_12]
        counts_12 = self.edge_counter[self.corr_12]
        self.t12 = self.t12 + edge_vals_12 / counts_12
        
        edge_vals_13 = self.edge_costs[self.corr_13]
        counts_13 = self.edge_counter[self.corr_13]
        self.t13 = self.t13 + edge_vals_13 / counts_13

        edge_vals_23 = self.edge_costs[self.corr_23]
        counts_23 = self.edge_counter[self.corr_23]
        self.t23 = self.t23 + edge_vals_23 / counts_23

        mask = self.edge_counter > 0
        self.edge_costs[mask] = 0.0

    def min_marginal(self, x, y, z):
        mm0 = torch.min(torch.stack([torch.zeros_like(y + z), y + z]))
        mm1 = torch.min(torch.stack([x + y + z, x + y, x + z]))
        return mm1 - mm0

    def send_messages_to_edges(self):
        for idx in range(self.t12.shape[0]):
            c12 = self.t12[idx]
            c13 = self.t13[idx]
            c23 = self.t23[idx]

            e12_diff = torch.zeros_like(c12)
            e13_diff = torch.zeros_like(c12)
            e23_diff = torch.zeros_like(c12)


            mm = self.min_marginal(c12, c13, c23)
            c12 -= (1.0 / 3.0) * mm
            e12_diff += (1.0 / 3.0) * mm

            mm = self.min_marginal(c13, c12, c23)
            c13 -= (1.0 / 2.0) * mm
            e13_diff += (1.0 / 2.0) * mm

            mm = self.min_marginal(c23, c12, c13)
            c23 -= mm
            e23_diff += mm

            mm = self.min_marginal(c12, c13, c23)
            c12 -= (1.0 / 2.0) * mm
            e12_diff += (1.0 / 2.0) * mm

            mm = self.min_marginal(c13, c12, c23)
            c13 -= mm 
            e13_diff += mm

            mm = self.min_marginal(c12, c13, c23)
            c12 -= mm
            e12_diff += mm

            self.t12[idx] = c12
            self.t13[idx] = c13
            self.t23[idx] = c23

            self.edge_costs[self.corr_12[idx]] += e12_diff
            self.edge_costs[self.corr_13[idx]] += e13_diff
            self.edge_costs[self.corr_23[idx]] += e23_diff

    def iteration(self):
        self.send_messages_to_triplets()
        self.send_messages_to_edges()


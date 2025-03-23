import torch

class ClassicalMessagePassing:
    def __init__(self, edge_costs, 
                 triangle_correspondence_12, triangle_correspondence_13, triangle_correspondence_23,
                 t12_costs, t13_costs, t23_costs,
                 edge_counter):
        
        self.edge_costs = edge_costs
        self.tri_corr_12 = triangle_correspondence_12
        self.tri_corr_13 = triangle_correspondence_13
        self.tri_corr_23 = triangle_correspondence_23
        self.t12_costs = t12_costs
        self.t13_costs = t13_costs
        self.t23_costs = t23_costs
        self.edge_counter = edge_counter

        assert all(t.device == edge_costs.device for t in [
            triangle_correspondence_12, triangle_correspondence_13, triangle_correspondence_23,
            t12_costs, t13_costs, t23_costs, edge_counter
        ])

    def compute_edge_lower_bound(self):
        return torch.sum(torch.where(self.edge_costs < 0, self.edge_costs, torch.zeros_like(self.edge_costs)))

    def compute_triangle_lower_bound(self):
        a = self.t12_costs
        b = self.t13_costs
        c = self.t23_costs
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
        for tri_costs, tri_corr in zip(
            [self.t12_costs, self.t13_costs, self.t23_costs],
            [self.tri_corr_12, self.tri_corr_13, self.tri_corr_23]
        ):
            edge_vals = self.edge_costs[tri_corr]
            counts = self.edge_counter[tri_corr].float().clamp(min=1) 
            tri_costs += edge_vals / counts

        mask = self.edge_counter > 0
        self.edge_costs[mask] = 0.0

    def min_marginal(self, x, y, z):
        mm0 = torch.min(torch.stack([torch.zeros_like(y + z), y + z]))
        mm1 = torch.min(torch.stack([x + y + z, x + y, x + z]))
        return mm1 - mm0

    def send_messages_to_edges(self):
        for idx in range(self.t12_costs.shape[0]):
            c12 = self.t12_costs[idx]
            c13 = self.t13_costs[idx]
            c23 = self.t23_costs[idx]

            e12_diff = torch.tensor(0.0, device=c12.device)
            e13_diff = torch.tensor(0.0, device=c12.device)
            e23_diff = torch.tensor(0.0, device=c12.device)

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

            self.t12_costs[idx] = c12
            self.t13_costs[idx] = c13
            self.t23_costs[idx] = c23

            self.edge_costs[self.tri_corr_12[idx]] += e12_diff
            self.edge_costs[self.tri_corr_13[idx]] += e13_diff
            self.edge_costs[self.tri_corr_23[idx]] += e23_diff

    def iteration(self):
        self.send_messages_to_triplets()
        print("[DEBUG] After triplet messages - edge costs:", self.edge_costs)
        print("[DEBUG] After triplet messages - t12/t13/t23:", self.t12_costs, self.t13_costs, self.t23_costs)

        self.send_messages_to_edges()

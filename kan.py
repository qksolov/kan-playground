import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KANLayer(torch.nn.Module):
    """
    KANLayer class

    Attributes:
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        grid_size (int): Number of grid points for B-splines.
        spline_order (int): Order of the B-splines.
        grid_range (list): Range of values for the grid, typically [-1, 1].
        grid (torch.Tensor): Grid used for B-spline computation.
        spline_weight (torch.nn.Parameter): Learnable B-spline weights.
        residual_fn (torch.nn.Module): Residual function b(x).
        residual_weight (torch.nn.Parameter): Learnable weights for the residual function.
        device (str): Device type ("cpu" or "cuda").
    """

    def __init__(
            self,
            in_dim,
            out_dim,
            grid_size=5,
            spline_order=3,
            grid_range=[-1, 1],
            residual_fn=torch.nn.SiLU,

    ):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        grid_step = (grid_range[1] - grid_range[0]) / grid_size
        self.grid = torch.linspace(
            grid_range[0] - grid_step * spline_order - 1e-4,
            grid_range[1] + grid_step * spline_order + 1e-4,
            grid_size + 1 + 2 * spline_order
        ).reshape(-1, 1)

        self.spline_weight = torch.nn.Parameter(
            torch.normal(0, 0.1, size=(out_dim, in_dim, grid_size + spline_order)))

        self.residual_fn = residual_fn()
        self.residual_weight = torch.nn.Parameter(torch.ones(out_dim, in_dim))

        self.device = "cpu"

    def to(self, device):
        super(KANLayer, self).to(device)
        self.grid = self.grid.to(device)
        self.device = device
        return self

    def b_splines(self, x):
        """
        Computes the B-splines for the input data x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, in_dim * (grid_size + spline_order)).
            Each row corresponds to a concatenation of the B-spline evaluations for each feature.

        """
        batch = x.shape[0]
        x = x.flatten()
        B = ((self.grid[:-1] <= x) & (x <= self.grid[1:])).to(torch.float32)
        for k in range(1, self.spline_order + 1):
            B = ((x - self.grid[: -k-1]) / (self.grid[k:-1] - self.grid[: -k-1]) * B[:-1]
                 + (self.grid[k + 1:] - x) / (self.grid[k + 1:] - self.grid[1:-k]) * B[1:])
        return B.T.reshape(batch, -1)

    def forward(self, x):
        # batch = x.shape[0]
        spline = torch.nn.functional.linear(
            self.b_splines(x), self.spline_weight.reshape(self.out_dim, -1))
        residual = torch.nn.functional.linear(
            self.residual_fn(x), self.residual_weight)
        x = spline + residual
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = torch.cat((self.spline_weight, self.residual_weight.unsqueeze(-1)), dim=2).abs().mean(-1)  # self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

    def node_scores(self):
        """
        Computes the node scores of layer

        Returns:
            tuple:
                torch.Tensor: Output scores for the input nodes (shape: (in_dim))
                torch.Tensor: Input scores for the output nodes (shape: (out_dim))
        """
        with torch.no_grad():
            l1_fake = self.spline_weight.abs().mean(-1)
            out_scores, _ = torch.max(l1_fake, dim=0)
            in_scores, _ = torch.max(l1_fake, dim=1)
            return out_scores, in_scores


class KAN(torch.nn.Module):
    def __init__(
            self,
            dims,
            grid_size=5,
            spline_order=3,
            grid_range=[-1, 1],
            residual_fn=torch.nn.SiLU,
    ):
        super(KAN, self).__init__()
        d = len(dims)
        self.layers = torch.nn.ModuleList()
        for i in range(d-1):
            self.layers.append(
                KANLayer(
                    dims[i],
                    dims[i+1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                    residual_fn=residual_fn,
                )
            )
        self.device = "cpu"

    def to(self, device):
        super(KAN, self).to(device)
        for layer in self.layers:
            layer.to(device)
        self.device = device
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(
                regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    def node_scores(self):
        """
        Compute the node scores for each layer in the model.

        Returns:
            list: A list of lists containing the node scores for the corresponding layer.
        """
        _, in_score = self.layers[0].node_scores()
        scores = []
        scores.append(torch.ones(
            self.layers[0].in_dim, device=in_score.device))

        for layer in self.layers[1:]:
            out_scores, next_in_scores = layer.node_scores()
            scores.append(torch.minimum(in_score, out_scores))
            in_score = next_in_scores

        scores.append(torch.ones(
            self.layers[-1].out_dim, device=in_score.device))

        scores = [score.cpu().tolist() for score in scores]
        return scores


def plot_KAN(model):
    """
    Visualization function KAN.  

    Specifically plots graphs of φ_i,j functions of each layer within the squares [-1,1]^2.  
    """

    def get_plot_points(layer, x):
        x = x.to(device)
        with torch.no_grad():
            spline = torch.nn.functional.linear(
                layer.b_splines(x),
                layer.spline_weight.reshape(-1, layer.grid_size + layer.spline_order)
            )
            residual = torch.nn.functional.linear(
                layer.residual_fn(x), layer.residual_weight.reshape(-1, 1))
            phi = (spline + residual).T.reshape(layer.out_dim, layer.in_dim, -1)
        return phi

    def color(score):
        color_value = 1 - max(min(float(score) / 0.03, 1.0), 0.2)
        return (color_value, color_value, color_value)

    plt.close()
    fig = plt.figure(figsize=(11.2, 8))
    with torch.no_grad():
        # To generalize, change the range of x using layer.grid_range inside the loop
        x = torch.linspace(-1, 1, 25).reshape(-1, 1)      

        node_scores = model.node_scores()
        plot_width, plot_height = 0.04, 0.056
        x_pos = np.linspace(plot_width, 1 - plot_width, 2 * len(model.layers) + 3)[1:-1]

        for layer_idx, layer in enumerate(model.layers):
            phi = get_plot_points(layer, x).cpu()

            y_in_pos = np.linspace(1, 0, layer.in_dim + 2)
            nodes_step = y_in_pos[0] - y_in_pos[1]
            plot_gap = nodes_step / layer.out_dim - plot_height

            spacing = 0 if layer.in_dim * layer.out_dim < 12 else 1.5 * plot_height
            y_plot_pos = np.linspace(
                y_in_pos[1] + nodes_step / 2 -
                plot_gap / 2 - plot_height + spacing,
                y_in_pos[-2] - nodes_step / 2 + plot_gap / 2 - spacing,
                layer.in_dim * layer.out_dim
            )

            y_out_pos = np.linspace(1, 0, layer.out_dim + 2)

            x_pos[2 * layer_idx + 1] -= plot_width / 2
            plot_idx = 0

            for i in range(layer.in_dim):
                for j in range(layer.out_dim):
                    color_value = color(
                        min(node_scores[layer_idx][i], node_scores[layer_idx + 1][j]))

                    ax = fig.add_axes(
                        [x_pos[2 * layer_idx + 1], y_plot_pos[plot_idx], plot_width, plot_height])
                    ax.plot(x, phi[j, i], color=color_value)
                    ax.spines[:].set_color(color_value)                    
                    ax.set(xticks=[], yticks=[], xlim=[-1, 1], ylim=[-1, 1]) # To generalize, change axis limits

                    # Lines connecting input nodes to plots
                    fig.add_artist(plt.Line2D(
                        [x_pos[2 * layer_idx], x_pos[2 * layer_idx + 1] - 0.005],
                        [y_in_pos[i + 1], y_plot_pos[plot_idx] + plot_height / 2],
                        transform=fig.transFigure, color=color_value
                    ))

                    # Lines connecting plots to output nodes
                    fig.add_artist(plt.Line2D(
                        [x_pos[2 * (layer_idx + 1)], x_pos[2 *
                                                           layer_idx + 1] + plot_width + 0.005],
                        [y_out_pos[j + 1], y_plot_pos[plot_idx] + plot_height / 2],
                        transform=fig.transFigure, color=color_value
                    ))

                    plot_idx += 1

                # Input layer nodes
                fig.add_artist(plt.Line2D(
                    [x_pos[2 * layer_idx]], [y_in_pos[i + 1]],
                    transform=fig.transFigure, color=color(
                        node_scores[layer_idx][i]),
                    marker="o", markersize=8, linestyle=""
                ))

        # Output node
        fig.add_artist(plt.Line2D(
            [x_pos[-1]], [0.5],
            transform=fig.transFigure, color="black",
            marker="o", markersize=8, linestyle=""
        ))

    return plt.gcf()

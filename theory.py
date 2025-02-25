text_1 = r"""<div style="text-align: justify !important; font-size: 16px !important;">

Welcome to the Kolmogorov-Arnold network playground. Here you can interactively configure
network parameters and observe the learning process in real time.

## What is Kolmogorov-Arnold network (KAN)?

Kolmogorov-Arnold networks were recently introduced in the paper [[1]](https://arxiv.org/abs/2404.19756) 
as an alternative to multilayer perceptrons (MLPs). They offer potential improvements in both accuracy and
interpretability.

The concept of KAN is rooted in the Kolmogorov-Arnold representation theorem 
[[2]](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem), which
states that any continuous function $ f : [0, 1]^n \to \mathbb{R} $ can be expressed as:

$$
f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right),
$$

where $ \phi_{q,p} : [0,1] \to \mathbb{R} $ and $ \Phi_q : \mathbb{R} \to \mathbb{R} $ are continuous functions.

This can be interpreted as a two-layer network with learnable activation functions. However,
since the theorem guarantees only the continuity of $ \phi_{q,p} $ and $ \Phi_q $, and not smoothness, they may
not be learnable in practice.

In the paper [[1]](https://arxiv.org/abs/2404.19756), the authors propose finding an approximate representation 
of the function $ f $ by increasing the depth and width of the layers. More specifically:

$$
\text{KAN}(\mathbf{x}) = \sum_{i_{L-1}=1}^{n_{L-1}} \phi_{L-1,i_L,i_{L-1}} \left( \sum_{i_{L-2}=1}^{n_{L-2}}
\cdots \left( \sum_{i_2=1}^{n_2} \phi_{2,i_3,i_2} \left( \sum_{i_1=1}^{n_1} \phi_{1,i_2,i_1} 
\left( \sum_{i_0=1}^{n_0} \phi_{0,i_1,i_0}(x_{i_0}) \right) \right)\right) \dots \right).
$$

The shape of such a KAN can be represented by the sequence $ [n_0, n_1, ..., n_L] $, where $ n_l $ is the 
number of nodes in the $ l^{\text{th}} $ layer.

</div>
"""


text_2 = r"""
<center>Figure 1: Notations of functions in the network. All functions are plotted in squares $ [-1,1]^2 $.</center>
<br />

<div style="text-align: justify !important; font-size: 16px !important;">
Each one-dimensional function $ \phi $ is represented as a linear combination of a set of basis B-splines $ B_i(x) $
and a residual function $ b(x) $:

$$
\phi = w_b b(x) + \sum_{i=1}^{N} w_i B_i(x),
$$

where $ N $ is determined by the spline order and the grid size.

Note that a spline is a piecewise polynomial function that ensures smooth transitions between its segments. 
Any spline of a fixed degree can be represented as a linear combination of B-splines of the same degree. 
To determine the basis B-splines, it is sufficient to specify the grid nodes. For more details, see 
[[3]](https://en.wikipedia.org/wiki/B-spline).
</div>

"""


text_3 = r"""
<center>Figure 2: B-splines of order 3 with a grid size of 5.</center>
<br />
<div style="text-align: justify; font-size: 16px;">

In most KAN implementations, the domains of the functions $ \phi_{i,j} $ can adapt during the learning
process by modifying or adding grid nodes, which changes the basis functions. Alternatively, the grid 
nodes can be made trainable, or layer normalization can be applied for better stability. For simplicity, 
in this playground, the basis functions remain fixed.

Another natural extension involves exploring different types of basis functions. The papers 
[[4]](https://arxiv.org/abs/2405.07200) and [[5]](https://arxiv.org/abs/2406.11173) discuss various 
approaches, among which are Radial Basis Functions and Chebyshev polynomials. These will be 
implemented here soon.

Additionally, the model can be simplified (by reducing the number of functions in the representation) 
through regularization. The playground implements the $ L_1 $ norms of functions $ \phi_{i,j} $ and 
layers $ \boldsymbol{\Phi} = (\phi_{i,j})_{i=1,j=1}^{n_{in},n_{out}} $, based on the idea described in 
[[6]](https://github.com/Blealtan/efficient-kan):

$$
|\phi|_1 = \frac{|w_b| + \sum |w_i|}{N + 1}, \qquad
|\boldsymbol{\Phi}|_1 = \sum_{i=1}^{n_{in}} \sum_{j=1}^{n_{out}} |\phi_{i,j}|_1.
$$

The entropy of the layer $ \boldsymbol{\Phi} $ is defined as

$$
S(\boldsymbol{\Phi}) = -\sum_{i=1}^{n_{in}} \sum_{j=1}^{n_{out}} \frac{|\phi_{i,j}|_1}{|\Phi|_1} \log \left( \frac{|\phi_{i,j}|_1}{|\Phi|_1} \right).
$$

Thus, the total loss is calculated as

$$
l_{\text{total}} = l_{\text{pred}} + \lambda_1 \sum_{l=0}^{L-1} |\boldsymbol{\Phi}_l|_1 + \lambda_2 \sum_{l=0}^{L-1} S(\boldsymbol{\Phi}_l),
$$

where $ \lambda_1 $ and $ \lambda_2 $ are regularization parameters, and $ l_{\text{pred}} $ denotes the prediction loss.

Below are some examples of toy functions with various KAN configurations that could be interesting to explore:

</div>
"""


references = r"""<div style="font-size: 16px;">

## References

[1] Kan: Kolmogorov-Arnold Networks, 2024. 
[arXiv:2404.19756](https://arxiv.org/abs/2404.19756)  

[2] Kolmogorov–Arnold Representation Theorem. 
[Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)  

[3] B-spline. [Wikipedia](https://en.wikipedia.org/wiki/B-spline)  

[4] Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear 
Function Approximation, 2024. [arXiv:2405.07200](https://arxiv.org/abs/2405.07200)  

[5] BSRBF-KAN: A Combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks, 
2024. [arXiv:2406.11173](https://arxiv.org/abs/2406.11173)  

[6] Efficient-KAN. [GitHub Repository](https://github.com/Blealtan/efficient-kan)

</div>
"""
$
\begin{aligned}
m &= \text{number of examples} \\
n &= \text{number of  inputs} \\
k &= \text{number of neurons}
\end{aligned}
$

## Calculating net input:

$$
\begin{aligned}
Z = X \cdot W + J \otimes B
\end{aligned}
$$

- $X$ is the input matrix with dimensions $m \times n$.
- $W$ is the weight matrix with dimensions $n \times k$
- $J$ is all ones with dimensions $m \times 1$
- $B$ is the bias matrix with dimensions $1 \times k$

The result $Z$ will have dimensions $m \times k$.

$$
\begin{aligned}
X &= \begin{bmatrix}
x_1^1 & x_2^1 & x_3^1 & \cdots & x_n^1 \\
x_1^2 & x_2^2 & x_3^2 & \cdots & x_n^2 \\
x_1^3 & x_2^3 & x_3^3 & \cdots & x_n^3 \\
\vdots & \vdots & \vdots & \ddots &  \vdots \\
x_1^m & x_2^m & x_3^m & \cdots & x_n^m \\
\end{bmatrix}_{m \times n}

\qquad

& W &= \begin{bmatrix}
w_1^1 & w_1^2 & w_1^3 & \cdots & w_1^k \\
w_2^1 & w_2^2 & w_2^3 & \cdots & w_2^k \\
w_3^1 & w_3^2 & w_3^3 & \cdots & w_3^k \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
w_n^1 & w_n^2 & w_n^3 & \cdots & w_n^k \\
\end{bmatrix}_{n \times k} \\


J &= \begin{bmatrix}
1 \\ 1\\ 1 \\ \vdots \\ 1
\end{bmatrix}_{m \times 1}

\qquad

& B &= \begin{bmatrix}
b_1 & b_2 & b_3 & \cdots & b_k \\
\end{bmatrix}_{1 \times k}
\end{aligned}
$$

**Note:** The kronecker product is not required to be explicitly implemented in code. As $B$ will be added to all rows of $W$ regardless.

## Gradients

### Gradient of the Error with respect to the activation of the last layer

The error functions is given by:

$$
E = \frac 1 m \sum^m_{i=0} \left (\boldsymbol {a}^{[L]}_i - \boldsymbol {y}_i \right )^2
$$

Where $\boldsymbol {a^{[L]}_i}$ is the activation of the last layer and $\boldsymbol {y}_i$ is the true output, over one example.

The gradient of the error with respect to the activations of the last layer is:

$$
\begin{aligned}
\frac{\partial E}{\partial \boldsymbol {a}^{[L]}_j}
&= \frac 1 m \cdot 2 \sum^m_{i=0} \left (\boldsymbol {a}^{[L]}_i - \boldsymbol {y}_i \right ) \cdot 1 \\

&= \frac 2 m \left (\boldsymbol {a}^{[L]}_j - \boldsymbol {y}_j \right ) \\
\end{aligned}
$$

Where $\boldsymbol {a^{[L]}_j}$ is one particular activation of the last layer over all examples.

Matrix notation of gradient of the error with respect to activations of the last layer is:

$$
\frac{\partial E}{\partial A^{[L]}}
= \frac 2 m \left (A^{[L]} - Y \right )
\qquad (\text{dimensions: }m \times k)
$$

Where $A^{[L]}$ is the activation of the last layer and $Y$ are true outputs, over all examples.

### Gradient of the activations of the last layer with respect to the net input of the last layer

$$
\frac {\partial A^{[L]}} {\partial Z^{[L]}}
= h^{\prime} (Z^{[L]})
\qquad (\text{dimensions: }m \times k)
$$

where $Z^{[L]}$ are the net inputs over all examples.

### Gradient of the net input of the last layer with respect to the activations of the previous layer

Given that:

$$
Z^{[L]} = A^{[L-1]} \cdot W^{[L]} + \boldsymbol{b}^{[L]}
$$

The partial derivative of $Z^{[L]}$ with respect to $A^{[L-1]}$ is:

$$
\frac {\partial Z^{[L]}} {\partial A^{[L-1]}}
= W^{[L]}
\qquad (\text{dimensions: }n \times k)
$$

### Gradient of the net input of the last layer with respect to the weights

The partial derivative of $Z^{[L]}$ with respect to $W^{[L]}$ is:

$$
\frac {\partial Z^{[L]}} {\partial w^{[L]}}
= A^{[L-1]}
\qquad (\text{dimensions: }m \times n)
$$

where $A^{[L - 1]}$ are the activations of previous layer over all examples.

### Gradient of the net input of the last layer wit respect to the biases

The partial derivative of $Z^{[L]}$ with respect to $\boldsymbol{b}^{[L]}$ is:

$$
\frac {\partial Z^{[L]}} {\partial \boldsymbol{b}^{[L]}}
= 1
$$

## Combining Gradients

combining these gradients using the chain rule:

To generalize the equations over all layers, $\partial(l)$ can be used in place of $\frac {\partial E}{\partial A^{[l]}}$ where $l$ indicates $l$-th Layer.

The gradient of the last layer $\partial (L)$ is given as,

$$
\partial (L) = \frac 2 m \left (A^{[L]} - Y \right )
$$

### Gradient with respect to the previous layer's activations

$$
\begin{aligned}
\partial (l - 1)
&= \left (\frac{\partial A^{[L]}}{\partial Z^{[L]}}
\times \partial (l) \right )
\cdot \left (\frac{\partial Z^{[L]}}{\partial A^{[L-1]}} \right )^T \\

&= \left [h^\prime (Z^{[L]}) \times \partial (l) \right ]
\cdot (W^{[L]})^T

\qquad (\text{dimensions: } m \times n )

\end{aligned}
$$

### Gradients with respect to the weights

$$
\begin{aligned}
\frac{\partial E}{\partial w^{[L]}}
&= \left (\frac{\partial Z^{[L]}}{\partial w^{[L]}} \right )^T
\cdot \left (\frac{\partial A^{[L]}}{\partial Z^{[L]}}
\times \partial (l) \right ) \\

&= (A^{[L-1]})^T
\cdot \left [h^\prime (Z^{[L]}) \times \partial (l) \right ]

\qquad (\text{dimensions: } n \times k )
\end{aligned}
$$

### Gradients with respect to the biases

$$
\begin{aligned}
\frac{\partial E}{\partial \boldsymbol{b}^{[L]}}
&= \frac{\partial Z^{[L]}}{\partial \boldsymbol{b}^{[L]}}
\cdot \left (\frac{\partial A^{[L]}}{\partial Z^{[L]}}
\times \partial (l) \right ) \\

\text{since }\frac {\partial Z^{[L]}} {\partial \boldsymbol{b}^{[L]}} = 1
\text{, we have } \\

&= \frac{\partial A^{[L]}}{\partial Z^{[L]}}
\times \partial (l) \\

&= h^\prime (Z^{[L]}) \times \partial (l)

\qquad (\text{dimensions: } m \times k )
\end{aligned}
$$

## Updating parameters

### Updating Weights

$$
W^{\prime [l]} = W - \alpha \frac {\partial E} {\partial W^{[L]}}
$$

### Updating Biases

The partial derivatives over all examples are added,

$$
\boldsymbol{b}^{\prime [l]} = b - \alpha \sum^m_{i=0} \frac {\partial E} {\partial b_i^{[L]}}
$$

Here $\alpha$ is the learning rate.

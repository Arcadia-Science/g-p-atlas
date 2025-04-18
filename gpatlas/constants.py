# define a minimum value for variables to avoid divide by zero and buffer underflow.
EPS = 1e-15

# TODO: consider using PReLU instead of LeakyReLU to avoid the need for this constant.
LEAKY_RELU_SLOPE = 0.01

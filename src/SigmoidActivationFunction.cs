using MathNet.Numerics.LinearAlgebra;

namespace neural_network
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        public Matrix<double> Activation(Matrix<double> input)
            => input.Map((x) => 1 / (1 + System.Math.Exp(-x)));

        public Matrix<double> ActivationPrime(Matrix<double> input)
            => input.Map((x) => x * (1 - x));
    }
}
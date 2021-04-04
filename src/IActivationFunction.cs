using MathNet.Numerics.LinearAlgebra;

namespace neural_network
{
    public interface IActivationFunction
    {
        Matrix<double> Activation(Matrix<double> input);
        Matrix<double> ActivationPrime(Matrix<double> input);
    }
}
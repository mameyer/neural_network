using System;
using MathNet.Numerics.LinearAlgebra;

namespace neural_network
{
    public class TanhActivationFunction : IActivationFunction
    {
        public Matrix<double> Activation(Matrix<double> input)
            => input.Map((x) => Math.Tanh(x));

        public Matrix<double> ActivationPrime(Matrix<double> input)
            => input.Map((x) => 1 - Math.Pow(Math.Tanh(x), 2));
    }
}
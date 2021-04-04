using System;
using MathNet.Numerics.LinearAlgebra;

namespace neural_network
{
    public class ActivationLayer : Layer
    {
        public ActivationLayer(IActivationFunction activationFunction)
            : base()
        {
            this.ActivationFunction = activationFunction;
        }

        public IActivationFunction ActivationFunction { get; }

        public Matrix<double> Input { get; set; }
        public Matrix<double> Output { get; set; }

        public override Matrix<double> ForwardPropagation(Matrix<double> inputData)
        {
            this.Input = inputData.Clone();
            this.Output = this.ActivationFunction.Activation(this.Input);
            return this.Output;
        }

        public override Matrix<double> BackwardPropagation(Matrix<double> outputError, double learningRate)
        {
            return this.ActivationFunction.ActivationPrime(this.Input).PointwiseMultiply(outputError);
        }
    }
}
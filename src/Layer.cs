using MathNet.Numerics.LinearAlgebra;

namespace neural_network
{
    public abstract class Layer
    {
        public Layer()
        {
        }

        public int InputSize { get; internal set; }
        public int OutputSize { get; internal set; }
        public int BatchSize { get; internal set; }

        public abstract Matrix<double> ForwardPropagation(Matrix<double> input);
        public abstract Matrix<double> BackwardPropagation(Matrix<double> outputErrror, double learningRate);
    }
}
using MathNet.Numerics.LinearAlgebra;

namespace neural_network.mnist
{
    public class Image
    {
        public Matrix<double> Label { get; set; }
        public Matrix<double> Data { get; set; }
    }
}
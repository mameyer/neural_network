using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace neural_network
{
    public class MSELossFunction : ILossFunction
    {
        public double Loss(Matrix<double> yTrue, Matrix<double> yPred)
            => yTrue.Subtract(yPred).Power(2).Enumerate().Average();

        public Matrix<double> LossPrime(Matrix<double> yTrue, Matrix<double> yPred)
            => yPred.Subtract(yTrue).Multiply(2).Divide(yTrue.Enumerate().Count());
    }
}
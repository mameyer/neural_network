using MathNet.Numerics.LinearAlgebra;

namespace neural_network
{
    public interface ILossFunction
    {
        double Loss(Matrix<double> yTrue, Matrix<double> yPred);
        Matrix<double> LossPrime(Matrix<double> yTrue, Matrix<double> yPred);
    }
}
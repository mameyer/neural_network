using MathNet.Numerics.LinearAlgebra;

namespace neural_network
{
    public class FCLayer : Layer
    {
        public FCLayer(int inputSize, int outputSize, int batchSize = 1)
            : base()
        {
            this.InputSize = inputSize;
            this.OutputSize = outputSize;
            this.BatchSize = batchSize;

            this.Input = Matrix<double>.Build.Dense(this.BatchSize, this.InputSize);
            this.Weights = Matrix<double>.Build.Random(this.InputSize, this.OutputSize).Subtract(0.5);
            this.Output = Matrix<double>.Build.Dense(this.BatchSize, this.OutputSize);
            this.Bias = Matrix<double>.Build.Dense(this.BatchSize, this.OutputSize);

            InitializeBias(this.BatchSize);
        }

        private void InitializeBias(int batchSize)
        {
           this.Bias = Matrix<double>.Build.Random(batchSize, this.OutputSize).Subtract(0.5); 
        }

        // [BatchSize, InputSize]
        public Matrix<double> Input { get; set; }

        // [InputSize, OutputSize]
        public Matrix<double> Weights { get; set; }

        // [BatchSize, OutputSize]
        public Matrix<double> Output { get; set; }
        
        // [BatchSize, OutputSize]
        public Matrix<double> Bias { get; set; }

        public override Matrix<double> ForwardPropagation(Matrix<double> input)
        {
            int batchSize = input.RowCount;
            if (batchSize != this.BatchSize)
            {
                throw new System.Exception($"invalid batch size: got {batchSize}, expected {this.BatchSize}");
            }

            // [BatchSize, InputSize]
            this.Input = input.Clone();

            // [BatchSize, OutputSize] = ([BatchSize, InputSize] * [InputSize, OutputSize]) + [BatchSize, OutputSize]
            this.Output = this.Input.Multiply(this.Weights).Add(this.Bias);

            // [BatchSize, OutputSize]
            return this.Output;
        }

        public override Matrix<double> BackwardPropagation(Matrix<double> outputError, double learningRate)
        {
            int batchSize = outputError.RowCount;
            if (batchSize != this.BatchSize)
            {
                throw new System.Exception($"invalid batch size: got {batchSize}, expected {this.BatchSize}");
            }

            // [BatchSize, InputSize] = [BatchSize, OutputSize] * [OutputSize, InputSize]
            var inputError = outputError.Multiply(this.Weights.Transpose());

            // [InputSize, OutputSize] = [InputSize, BatchSize] * [BatchSize, OutputSize]
            var weightsError = this.Input.Transpose().Multiply(outputError);

            // [InputSize, OutputSize] = ([InputSize, BatchSize] - [InputSize, BatchSize]) * f
            this.Weights = this.Weights.Subtract(weightsError.Multiply(learningRate));

            // [BatchSize, OutputSize] = [BatchSize, OutputSize] - ([BatchSize, OutputSize] * f)
            this.Bias = this.Bias.Subtract(outputError.Multiply(learningRate));

            // [BatchSize, InputSize]
            return inputError;
        }
    }
}
using System.Collections;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;
using System;

namespace neural_network.mnist
{
    public static class MnistLoader
    {
        private const string basePath = "mnist";
        private const string trainImagesFilename = "train-images-idx3-ubyte";
        private const string trainLabelsFilename = "train-labels-idx1-ubyte";
        private const string testImagesFilename = "t10k-images-idx3-ubyte";
        private const string testLabelsFilename = "mnist/t10k-labels-idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(basePath + "/" + trainImagesFilename, basePath + "/" + trainLabelsFilename))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(basePath + "/" + testImagesFilename, basePath + "/" + testLabelsFilename))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var imageDataBytes = images.ReadBytes(width * height);
                var data = Matrix<double>.Build.Dense(1, imageDataBytes.Length);
                for (int j = 0; j < imageDataBytes.Length; j++)
                {
                    data[0, j] = Convert.ToDouble(imageDataBytes[j]) / 255.0;
                }

                var labelDataBytes = labels.ReadByte();
                Matrix<double> label = Matrix<double>.Build.Dense(1, 10, 0);
                int labelPos = Convert.ToInt32(labelDataBytes);
                label[0, labelPos] = 1;

                yield return new Image()
                {
                    Data = data,
                    Label = label
                };
            }
        }
    }
}
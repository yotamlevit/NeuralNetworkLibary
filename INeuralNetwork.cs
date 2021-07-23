using GraphLibary;
using System;

namespace NeuralNetworkLibary
{

    public interface INeuralNetwork
    {
        //Update the NN same as constactor
        //1) dGraph - a Directed Graph that will represent the NN
        //2) layerCount - an array in length of the number of layers in the NN/ the graph
        //   
        void UpdateNeuralNetwork(DirectedGraph<string, double> dGraph, int[] layerCount);

        //return the output from the all NN with a given input
        //1)inputs - an array if doubles that contain the givin inputs
        int NextMove(double[] inputs);

        //print the metrixes
        void PrintMetrix();
    }
}

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

/**
 *
 * @author chloe
 */

import java.io.File;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.evaluation.classification.Evaluation;

public class PremiereIANeuralNetworkMNIST {
    static MultiLayerNetwork model;
    
      
        public static void main(String[] args) throws Exception {
            int batchSize = 128; // nombre d'exemples par lot pour l'entrainement
            int seed = 123; // initialisation du générateur de nombres aléatoires, afin d'assurer la reproductibilité
            int numEpochs = 10; //nombre de passes complètes sur le jeu de données d'entrainement
            double learningRate = 0.001; // taux d'apprentissage utilisé par l
            DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true,seed);
            DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false,seed);
            DataNormalization scaler = new NormalizerMinMaxScaler(0, 1);
            scaler.fit(mnistTrain);
            mnistTrain.setPreProcessor(scaler);
            mnistTest.setPreProcessor(scaler);
       
            
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate, 0.9))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                    .nIn(28 * 28)
                    .nOut(256)
                    .activation(Activation.RELU)
                    .build())
                .layer(new DenseLayer.Builder()
                    .nOut(256)
                    .activation(Activation.RELU)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(10)
                    .activation(Activation.SOFTMAX)
                    .build())
                .build();
        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        System.out.println("Training model...");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(mnistTrain);
            System.out.println("Completed epoch " + (i + 1));
        }
        System.out.println("Evaluating model...");
        Evaluation eval = model.evaluate(mnistTest);
        System.out.println(eval.stats());

        model.save(new File("mnist-mlp-model.zip"));

        System.out.println("Model training complete.");
}
}
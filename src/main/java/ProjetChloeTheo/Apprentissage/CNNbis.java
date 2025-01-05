/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

/**
 *
 * @author chloe
 */

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import java.util.Collections; 

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

public class CNNbis {
    private static MultiLayerNetwork model;
    
    public static void main(String[] args) {
        try {
            String csvFilePath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba8000OMCNN-OPPER.csv";
            
            // Paramètres du modèle
            int seed = 123;
            double learningRate = 0.001;
            int numEpochs = 30;
            int batchSize = 128;  // Ajout de la taille du batch
            
            // Création du dataset
            DataSet fullDataset = createDataset(csvFilePath);
            System.out.println("Dataset créé avec " + fullDataset.numExamples() + " exemples.");
            
            // Division en ensembles d'entraînement et de test
            DataSet[] splits = splitDataset(fullDataset, 0.8);
            
            // Création des iterators avec batch
            DataSetIterator trainIterator = new ListDataSetIterator<>(splits[0].asList(), batchSize);
            DataSetIterator testIterator = new ListDataSetIterator<>(splits[1].asList(), batchSize);
            
            // Création du modèle
            model = createModel(seed, learningRate);
            
            // Entraînement du modèle avec batch
            System.out.println("Starting training...");
            trainModelWithBatch(model, trainIterator, numEpochs);
            
            // Sauvegarde du modèle
            String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn2-model.zip";
            saveModel(model, modelPath);
            
            // Évaluation du modèle
            System.out.println("\nÉvaluation du modèle...");
            evaluateModel(modelPath, testIterator);
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    // Nouvelle méthode d'entraînement avec batch
    private static void trainModelWithBatch(MultiLayerNetwork model, DataSetIterator trainIterator, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            trainIterator.reset();
            int batchNum = 0;
            while (trainIterator.hasNext()) {
                DataSet batch = trainIterator.next();
                model.fit(batch);
                if (batchNum % 10 == 0) {
                    System.out.printf("Epoch %d, Batch %d: Score = %.4f%n", 
                        epoch + 1, batchNum, model.score());
                }
                batchNum++;
            }
            System.out.println("Completed epoch " + (epoch + 1));
        }
    }
    
    // Méthode qui évalue les performances du modèle
    public static void evaluateModel(String modelPath, DataSetIterator testIterator) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelPath));
        System.out.println("Modèle chargé depuis: " + modelPath);

        RegressionEvaluation eval = new RegressionEvaluation();
        int totalPredictions = 0;
        int correctPredictions = 0;
        double mseSum = 0.0;
        double threshold = 0.5;

        testIterator.reset();
        while (testIterator.hasNext()) {
            DataSet batch = testIterator.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            
            if (features.rank() != 4) {
                features = features.reshape(features.size(0), 1, 8, 8);
            }
            
            INDArray predictions = model.output(features);
            eval.eval(labels, predictions);
            
            for (int i = 0; i < predictions.length(); i++) {
                totalPredictions++;
                double predicted = predictions.getDouble(i);
                double actual = labels.getDouble(i);
                
                if ((predicted >= threshold && actual >= threshold) ||
                    (predicted < threshold && actual < threshold)) {
                    correctPredictions++;
                }
                
                mseSum += Math.pow(predicted - actual, 2);
            }
        }
        
        // Calcul et affichage des métriques
        double accuracy = (double) correctPredictions / totalPredictions * 100;
        double mse = mseSum / totalPredictions;
        double rmse = Math.sqrt(mse);
        
        System.out.println("\nRésultats de l'évaluation du modèle CNN :");
        System.out.println("----------------------------------------");
        System.out.printf("Précision de classification : %.2f%% (%d/%d)%n", 
                accuracy, correctPredictions, totalPredictions);
        System.out.printf("Erreur quadratique moyenne (MSE) : %.4f%n", mse);
        System.out.printf("RMSE : %.4f%n", rmse);
        System.out.printf("Coefficient de corrélation : %.4f%n", eval.pearsonCorrelation(0));
        System.out.printf("MAE : %.4f%n", eval.averageMeanAbsoluteError());
        
        // Recommendations
        System.out.println("\nRecommandations :");
        if (accuracy < 60) {
            System.out.println("- Considérez l'augmentation de la taille du batch");
            System.out.println("- Ajustez le learning rate");
        } else if (accuracy > 95) {
            System.out.println("- Surveillez le surapprentissage");
            System.out.println("- Vous pouvez essayer de réduire la taille du batch");
        }
    }
    
    // Création du dataset spécifique pour CNN (reshape des données en 8x8)
    public static DataSet createDataset(String csvFilePath) throws IOException {
        List<INDArray> inputList = new ArrayList<>();
        List<INDArray> outputList = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length != 65) continue;
                
                // Création d'un tableau 8x8x1 pour l'entrée CNN
                INDArray input = Nd4j.zeros(1, 1, 8, 8);
                for (int i = 0; i < 64; i++) {
                    int row = i / 8;
                    int col = i % 8;
                    input.putScalar(new int[]{0, 0, row, col}, Double.parseDouble(values[i]));
                }
                
                // Sortie (probabilité de victoire)
                INDArray output = Nd4j.zeros(1, 1);
                output.putScalar(0, Double.parseDouble(values[64]));
                
                inputList.add(input);
                outputList.add(output);
            }
        }
        
        // Concaténation des données
        INDArray input = Nd4j.vstack(inputList);
        INDArray output = Nd4j.vstack(outputList);
        
        DataSet dataset = new DataSet(input, output);
        
        // Normalisation
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataset);
        normalizer.transform(dataset);
        
        return dataset;
    }
    
    // Création du modèle CNN
    public static MultiLayerNetwork createModel(int seed, double learningRate) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(learningRate))
            .list()
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nIn(1)
                .nOut(64)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(1, 1)
                .build())
            .layer(new ConvolutionLayer.Builder(2, 2)
                .nOut(128)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new ConvolutionLayer.Builder(2, 2)
                .nOut(128)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new DenseLayer.Builder()
                .nOut(256)
                .activation(Activation.RELU)
                .dropOut(0.5)  // Ajout de dropout pour éviter le surapprentissage
                .build())
            .layer(new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .dropOut(0.3)  // Ajout de dropout pour éviter le surapprentissage
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nOut(1)
                .activation(Activation.IDENTITY)
                .build())
            .setInputType(InputType.convolutional(8, 8, 1))
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        
        return model;
    }
    
    public static void saveModel(MultiLayerNetwork model, String modelPath) throws IOException {
        System.out.println("Saving the model...");
        ModelSerializer.writeModel(model, new File(modelPath), true);
        System.out.println("Model saved to " + modelPath);
    } 
    
    // Méthode pour diviser le dataset en ensembles d'entraînement et de test
    public static DataSet[] splitDataset(DataSet fullDataset, double trainRatio) {
        fullDataset.shuffle();
        int numExamples = fullDataset.numExamples();
        int trainSize = (int) (numExamples * trainRatio);
        
        SplitTestAndTrain splitSets = fullDataset.splitTestAndTrain(trainSize);
        return new DataSet[]{splitSets.getTrain(), splitSets.getTest()};
    }
    
    
}

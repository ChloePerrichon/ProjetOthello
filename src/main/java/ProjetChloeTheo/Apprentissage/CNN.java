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

public class CNN {
    private static MultiLayerNetwork model; // Déclaration du modèle CNN
    
    public static void main(String[] args) {
        try {
            String csvFilePath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba8000OMCNN-OPPER.csv"; // fichier de données csv
            
            // Paramètres du modèle
            int seed = 123; // nombre de reproductibilité
            double learningRate = 0.001; // taux d'apprentissage
            int numEpochs = 30; //nombre d'époque
            int batchSize = 128;  //  taille du batch
            
            // Création du dataset à partir du csv
            DataSet fullDataset = createDataset(csvFilePath);
            System.out.println("Dataset créé avec " + fullDataset.numExamples() + " exemples.");
            
            // Division en ensembles d'entraînement et de test
            DataSet[] splits = splitDataset(fullDataset, 0.8);
            
            // Création des itérateurs de données 
            DataSetIterator trainIterator = new ListDataSetIterator<>(splits[0].asList(), batchSize); // données pour l'entrainement
            DataSetIterator testIterator = new ListDataSetIterator<>(splits[1].asList(), batchSize); // données pour les tests
            
            // Création du modèle
            model = createModel(seed, learningRate);
            
            // Entraînement du modèle 
            System.out.println("Starting training...");
            trainModel(model, trainIterator, numEpochs);
            
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
    
    // Entraine le modèle avec des lots de données
    private static void trainModel(MultiLayerNetwork model, DataSetIterator trainIterator, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            trainIterator.reset(); // remet à zéro pour chaque époque
            int batchNum = 0;
            while (trainIterator.hasNext()) {
                DataSet batch = trainIterator.next(); // charge le batch
                model.fit(batch); // entraine le modèle sur le batch
                if (batchNum % 10 == 0) {
                    System.out.printf("Epoch %d, Batch %d: Score = %.4f%n", 
                        epoch + 1, batchNum, model.score()); // affiche le score tous les 10 batchs
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
        double threshold = 0.5; // seuil de classification

        testIterator.reset();
        while (testIterator.hasNext()) {
            DataSet batch = testIterator.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            
            if (features.rank() != 4) {
                features = features.reshape(features.size(0), 1, 8, 8);
            }
            
            INDArray predictions = model.output(features); // prédictions du modèle
            eval.eval(labels, predictions); // évaluation des prédictions
            
            for (int i = 0; i < predictions.length(); i++) {
                totalPredictions++;
                double predicted = predictions.getDouble(i);
                double actual = labels.getDouble(i);
                
                if ((predicted >= threshold && actual >= threshold) ||
                    (predicted < threshold && actual < threshold)) {
                    correctPredictions++;
                }
                
                mseSum += Math.pow(predicted - actual, 2); // somme des erreurs quadratiques
            }
        }
        
        // Calcul et affichage des métriques de performance
        double accuracy = (double) correctPredictions / totalPredictions * 100;
        double mse = mseSum / totalPredictions;
        double rmse = Math.sqrt(mse);
        
        // affiche les résultats
        System.out.println("\nRésultats de l'évaluation du modèle CNN :");
        System.out.println("----------------------------------------");
        System.out.printf("Précision de classification : %.2f%% (%d/%d)%n", 
                accuracy, correctPredictions, totalPredictions);
        System.out.printf("Erreur quadratique moyenne (MSE) : %.4f%n", mse);
        System.out.printf("RMSE : %.4f%n", rmse);
        System.out.printf("Coefficient de corrélation : %.4f%n", eval.pearsonCorrelation(0));
        System.out.printf("MAE : %.4f%n", eval.averageMeanAbsoluteError());
        
    }
    
    // Création du dataset à partir du fichier CSVProba avec des matrices 8x8
    public static DataSet createDataset(String csvFilePath) throws IOException {
        List<INDArray> inputList = new ArrayList<>();
        List<INDArray> outputList = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length != 65) continue; //ignore les lignes qui ont plus de 65 colonnes
                
                // Création d'un tableau 8x8 pour l'entrée CNN
                INDArray input = Nd4j.zeros(1, 1, 8, 8);
                for (int i = 0; i < 64; i++) {
                    int row = i / 8;
                    int col = i % 8;
                    input.putScalar(new int[]{0, 0, row, col}, Double.parseDouble(values[i]));
                }
                
                // Sortie qui correspond à la probabilité
                INDArray output = Nd4j.zeros(1, 1);
                output.putScalar(0, Double.parseDouble(values[64]));
                
                inputList.add(input);
                outputList.add(output);
            }
        }
        
        
        INDArray input = Nd4j.vstack(inputList); // concaténation des entrées
        INDArray output = Nd4j.vstack(outputList); // concaténation des sorties
        
        DataSet dataset = new DataSet(input, output);
        
        // Normalisation de données
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataset);
        normalizer.transform(dataset);
        
        return dataset;
    }
    
    // Création de l'achitecture du modèle CNN
    public static MultiLayerNetwork createModel(int seed, double learningRate) {
        //Configuration du modèle CNN
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed) // reproduit les résultats
            .weightInit(WeightInit.XAVIER) //initialise les poids
            .updater(new Adam(learningRate)) // ajuste les poids 
            .list()
            // première couche de convolution
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nIn(1) // 1 entrée
                .nOut(64) // 64 filtres
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
             //  couche de pooling
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(1, 1)
                .build())
             // deuxième couche de convolution
            .layer(new ConvolutionLayer.Builder(2, 2)
                .nOut(128)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
             // troisième couche de convolution
            .layer(new ConvolutionLayer.Builder(2, 2)
                .nOut(128)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
             // couche dense
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
                
             //couche de sortie   
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nOut(1)
                .activation(Activation.IDENTITY)
                .build())
            .setInputType(InputType.convolutional(8, 8, 1))
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init(); // intialisation des paramètres du modèle
        model.setListeners(new ScoreIterationListener(10)); // affiche le score toutes les 10 itérations
        
        return model;
    }
    
    //methode qui permet de sauvegarder le modèle entrainé
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

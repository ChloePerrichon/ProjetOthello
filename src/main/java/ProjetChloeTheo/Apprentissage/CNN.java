/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

/**
 *
 * @author chloe
 */

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.util.ModelSerializer;
public class CNN {
    private static MultiLayerNetwork model;
     
    public static void main(String[] args) {
        try {
            String csvFilePath = "C:\\temp\\noirsProba10000.csv"; // Chemin vers votre fichier CSV

            DataSet dataset = createDataset(csvFilePath);
            System.out.println("Dataset created with " + dataset.numExamples() + " examples.");

            // Split du dataset en ensembles d'entraînement et de test
            dataset.shuffle();
            int trainSize = (int) (dataset.numExamples() * 0.8);
            int testSize = dataset.numExamples() - trainSize;
            DataSet trainData = dataset.splitTestAndTrain(trainSize).getTrain();
            DataSet testData = dataset.splitTestAndTrain(testSize).getTest();

            DataSetIterator trainIterator = new ListDataSetIterator<>(trainData.asList(), 128);
            DataSetIterator testIterator = new ListDataSetIterator<>(testData.asList(), 128);

            // Paramètres du modèle
            int seed = 123;
            double learningRate = 0.001;
            int numEpoque = 30;

            // Création du modèle
            model = createModel(seed, learningRate);

            // Entraînement du modèle
            trainModel(model, trainIterator,numEpoque);

            // Enregistrement du modèle
            saveModel(model, "othello-mlp-model-oraclepondere-noir.zip");
            
            // Évaluation du modèle
            //String modelPath = "C:\\Users\\chloe\\Desktop\\ProjetOthello\\othello-mlp-model-oraclepondere-noir.zip";
            //evaluateModel(modelPath, testIterator);

            // Exemple de nouvelle entrée pour faire une prédiction
            //INDArray newInput = Nd4j.create(new double[]{/* valeurs de l'entrée */}, new int[]{1, 64});
            //makePrediction(model, newInput);

        } catch (IOException | IllegalArgumentException e) {
            e.printStackTrace();
        }
    }
     
    // Fonction de création du dataset à partir d'un fichier CSV
    public static DataSet createDataset(String csvFilePath) throws IOException {
        List<INDArray> inputList = new ArrayList<>();
        List<INDArray> outputList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            int lineNumber = 0;
            while ((line = br.readLine()) != null) {
                lineNumber++;
                String[] values = line.split(",");
                if (values.length != 65) { // Ajusté pour 65 colonnes (64 entrées + 1 sortie)
                    System.err.println("Line " + lineNumber + " has " + values.length + " columns: " + line);
                    continue; // Sauter cette ligne
                }

                // Créer l'entrée INDArray avec 64 valeurs
                INDArray input = Nd4j.zeros(1, 64);
                boolean validInput = true;
                for (int i = 0; i < 64; i++) {
                    try {
                        input.putScalar(i, Double.parseDouble(values[i]));
                    } catch (NumberFormatException e) {
                        System.err.println("Invalid number at line " + lineNumber + " column " + i + ": " + values[i]);
                        validInput = false;
                        break;
                    }
                }
                if (!validInput) continue;

                // Créer la sortie INDArray avec 1 valeur
                INDArray output = Nd4j.zeros(1, 1);
                try {
                    output.putScalar(0, Double.parseDouble(values[64]));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid number at line " + lineNumber + " column 64: " + values[64]);
                    continue;
                }
                
                // Ajouter les entrées et sorties validées à leurs listes respectives
                inputList.add(input);
                outputList.add(output);
            }
        }
        
        if (inputList.isEmpty() || outputList.isEmpty()) {
            throw new IllegalArgumentException("No valid data found in the CSV file.");
        }

        // Empiler les entrées et sorties en un seul INDArray
        INDArray input = Nd4j.vstack(inputList);
        INDArray output = Nd4j.vstack(outputList);
        
        // Vérification de la forme des tableaux
        if (input.rank() != 2 || output.rank() != 2) {
            throw new IllegalStateException("Input or output has invalid shape. Rank: " + input.rank() + ", " + output.rank());
        }

        DataSet dataset = new DataSet(input, output);

        // Normalisation du dataset
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataset);
        normalizer.transform(dataset);

        return dataset;
    }
    
    // Fonction de création du modèle de réseau neuronal
    public static MultiLayerNetwork createModel(int seed, double learningRate) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(learningRate, 0.9))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(64) // Nombre d'entrées (64 cases du plateau Othello)
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new DenseLayer.Builder()
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Utilisation de la MSE pour la régression
                .nOut(1) // Une seule sortie pour la probabilité de victoire
                .activation(Activation.IDENTITY) // Activation identité pour la régression
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        return model;
    }
    
    // Fonction d'entraînement du modèle
    public static void trainModel(MultiLayerNetwork model, DataSetIterator trainIterator,int numEpoque) {
        System.out.println("Training model...");
        for (int i = 0; i < numEpoque; i++) { // Nombre d'époques = 30
            model.fit(trainIterator);
            System.out.println("Completed epoch " + (i + 1));
        }
    }
    
    // Fonction d'enregistrement du modèle
    public static void saveModel(MultiLayerNetwork model, String modelPath) throws IOException {
        System.out.println("Saving the model...");
        model.save(new File(modelPath));
        System.out.println("Model saved to " + modelPath);
    }
    
    // Fonction pour charger un modèle enregistré à partir d'un fichier ZIP
    public static MultiLayerNetwork loadModel(String modelPath) throws IOException {
        System.out.println("Loading model from: " + modelPath);
        MultiLayerNetwork loadedModel = MultiLayerNetwork.load(new File(modelPath), true);
        System.out.println("Model loaded successfully.");
        return loadedModel;
    }
    
    // Fonction d'évaluation du modèle
    public static void evaluateModel(String modelPath, DataSetIterator testIterator) throws IOException {
        // Charger le modèle à partir du fichier donné (par son chemin)
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
        System.out.println("Evaluating model...");
        // Créer une instance d'évaluation
        RegressionEvaluation eval = new RegressionEvaluation();
        // Parcourir les données de test pour faire des prédictions et évaluer le modèle
        while (testIterator.hasNext()) {
            DataSet t = testIterator.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);  // Obtenez les prédictions du modèle
            eval.eval(labels, predicted);  // Comparez les prédictions aux labels réels
        }
        // Afficher les résultats d'évaluation
        System.out.println(eval.stats());
    }
    
    // Exemple pour utiliser le modèle pour faire une prédiction avec de nouvelles données
    public static void makePrediction(MultiLayerNetwork model, INDArray newInput) {
        // Assurez-vous que `newInput` a les bonnes dimensions
        if (newInput.shape()[1] != 64) {
            throw new IllegalArgumentException("L'entrée doit avoir 64 colonnes.");
        }

        // Prédiction avec le modèle
        INDArray output = model.output(newInput);
        double prediction = output.getDouble(0);

        // Affichage de la prédiction
        System.out.printf("Prédiction: %.2f%%\n", prediction * 100);
    }
}
    


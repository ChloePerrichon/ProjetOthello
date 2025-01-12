/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage.modeles;

/**
 *
 * @author chloe
 */

import ProjetChloeTheo.Apprentissage.database.DataBaseEnvironment;
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
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

public class CNN {
    private static MultiLayerNetwork model; // Déclaration du modèle CNN
    
    public static void main(String[] args) {
        try {
            String csvFilePath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvSansEntrainementProba\\noirsProba5000.csv"; // chemin du csv sur lequel on veut entrainer le modèle
            
            // Paramètres du modèle
            int seed = 123; // graine de reproductibilité
            double learningRate = 0.001; // taux d'apprentissage
            int numEpochs = 30; //nombre d'époques : nombre de fois que le modèle voit l'ensemble des données
            int batchSize = 128;  //  taille du batch
            
            // Création du dataset à partir du csv
            DataSet fullDataset = createDataset(csvFilePath); // On crée un dataset en INDArray à partir du csv contenant les situations et les probas
            System.out.println("Dataset créé avec " + fullDataset.numExamples() + " exemples.");
            
            /// Division du dataset en deux ensembles: un d'entrainement et un de test
            DataSet[] splits = splitDataset(fullDataset, 0.8); 
            
            // Création des itérateurs de données 
            DataSetIterator trainIterator = new ListDataSetIterator<>(splits[0].asList(), batchSize); // création d'un itérateur pour parcourir les données d'entrainement par lots de taille batchSize (64)
            DataSetIterator testIterator = new ListDataSetIterator<>(splits[1].asList(), batchSize); // création d'un itérateur pour parcourir les données de test par lots de taille batchSize (128)
            
            // Création du modèle
            model = createModel(seed, learningRate);
            
            // Entraînement du modèle 
            System.out.println("Starting training...");
            trainModel(model, trainIterator, numEpochs);
            
            // Sauvegarde du modèle
            String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn3-model.zip";
            saveModel(model, modelPath);
            
            // Connection à la base de données
            DataBaseEnvironment.connect();
            
            // Export du modèle vers la base de données
            DataBaseEnvironment.exporterZIPversDatabase(modelPath, "Model");
            
            // Évaluation du modèle
            System.out.println("\nÉvaluation du modèle...");
            evaluateModel(modelPath, testIterator);
            
            // Fermeture de la connexion
            DataBaseEnvironment.close();
            
            /*//Entrainement d'un modèle déjà entrainé
            String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn3-model.zip";
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelPath)); // charge le modèle
            System.out.println("Modèle chargé depuis: " + modelPath);
            System.out.println("Starting training...");
            trainModel(model, trainIterator, numEpochs);*/ // Entraine le modèle
            
        } catch (IOException e) {
            e.printStackTrace(); // gestion des erreurs
        }
    }
    
    // Entraine le modèle avec des lots de données
    private static void trainModel(MultiLayerNetwork model, DataSetIterator trainIterator, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) { // Effectue l'entrainement sur le nombre d'époques demandées
            trainIterator.reset(); // Initialise l'itérateur au début des données
            int batchNum = 0; // Compteur pour voir la progression des lots traités
            while (trainIterator.hasNext()) { // Traitement des données par lots
                DataSet batch = trainIterator.next(); // Prends le prochain lot de données
                model.fit(batch); // Entraine le modèle sur le lot
                if (batchNum % 10 == 0) { // Boucle qui permet d'afficher la progression de l'entrainement
                    System.out.printf("Epoch %d, Batch %d: Score = %.4f%n", 
                        epoch + 1, batchNum, model.score()); // affiche le score tous les 10 batchs
                }
                batchNum++; // indente le compteur
            }
            System.out.println("Completed epoch " + (epoch + 1));
        }
    }
    
    // Méthode qui évalue les performances du modèle
    public static void evaluateModel(String modelPath, DataSetIterator testIterator) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelPath)); // Charge le modèle à évaluer
        System.out.println("Modèle chargé depuis: " + modelPath);

        RegressionEvaluation eval = new RegressionEvaluation(); // crée "eval" qui servira à évaluer les performances du modèle avec la bibliothèque DL4J
        int totalPredictions = 0; // création d'un compteur du nombre de prédiction
        int correctPredictions = 0; // création d'un compteur du nombre de prédiction correcte
        double mseSum = 0.0; // compteur correspondant à la somme des MSE calculées
        double threshold = 0.5; // seuil de classification
        
        //Boucle d'évaluation
        testIterator.reset(); // initialise l'itérateur de test au début des données
        while (testIterator.hasNext()) {
            DataSet batch = testIterator.next(); // prend le prochain lot de données
            INDArray features = batch.getFeatures(); // Données d'entrées que le modèle utilise pour faire les prédictions  
            INDArray labels = batch.getLabels(); // Valeurs réelles que l'on veut c'est à dire la 65eme valeur du dataset (la probabilité)
            
            //verifie la forme des données d'entrée et si nécessaire redimmensionne en une matrice 8x8
            if (features.rank() != 4) {
                features = features.reshape(features.size(0), 1, 8, 8);
            }
            
            INDArray predictions = model.output(features); // prédictions du modèle
            eval.eval(labels, predictions); // évaluation des prédictions avec la bibliothèque DL4J
            
            // calcule des métriques 
            for (int i = 0; i < predictions.length(); i++) {
                totalPredictions++; // indente le compteur des prédiction
                double predicted = predictions.getDouble(i);
                double actual = labels.getDouble(i);
                
                //vérifie si la prédiction est correcte selon le seuil 0.5
                if ((predicted >= threshold && actual >= threshold) ||
                    (predicted < threshold && actual < threshold)) {
                    correctPredictions++; // indente le compteur des prédictions correctes
                }
                
                mseSum += Math.pow(predicted - actual, 2); // calcule de l'erreur quadratique et ajoute la valeur à la somme des MSE
            }
        }
        
        
        double accuracy = (double) correctPredictions / totalPredictions * 100; // Précision du modèle
        double mse = mseSum / totalPredictions; // Erreur quadratique moyenne
        double rmse = Math.sqrt(mse); // Racine de l'erreur quadratique moyenne
        
        
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
        List<INDArray> inputList = new ArrayList<>(); // création d'une liste INDArray (DL4J) correspondant aux matrices 8x8 entrées 
        List<INDArray> outputList = new ArrayList<>(); // Liste INdArray correspondant aux sorties
        
        //Lecture du fichier Csv ligne par ligne
        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(","); // divise la ligne du csv pour récupérer les valeurs séparées par des virgules et donne un tableau de chaine qui contient les valeurs extraites
                if (values.length != 65) continue; //ignore les lignes qui ont plus de 65 colonnes
                
                // Création d'un tableau 8x8 pour l'entrée CNN
                INDArray input = Nd4j.zeros(1, 1, 8, 8); // initialisation d'un tableau 8x8 rempli de zéros
                for (int i = 0; i < 64; i++) { //parcours les 64 colonnes de la ligne
                    // calcule le positionnement de la valeur dans la ligne pour la placer correctement dans la matrice
                    int row = i / 8; 
                    int col = i % 8; 
                    input.putScalar(new int[]{0, 0, row, col}, Double.parseDouble(values[i])); // place les valeurs dans les positions correspondantes de la matrices 8x8
                }
                
                // Sortie qui correspond à la probabilité
                INDArray output = Nd4j.zeros(1, 1); 
                output.putScalar(0, Double.parseDouble(values[64]));
                
                // Ajoute les entrées et sorties validées à leurs listes respectives
                inputList.add(input);
                outputList.add(output);
            }
        }
        
        
        INDArray input = Nd4j.vstack(inputList); // concaténation des entrées
        INDArray output = Nd4j.vstack(outputList); // concaténation des sorties
        
              
        DataSet dataset = new DataSet(input, output);
        dataset.shuffle(); // mélange les données de manière aléatoire
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
            .layer(new ConvolutionLayer.Builder(3, 3) //filtre 3x3
                .nIn(1) // 1 entrée matrice 8x8
                .nOut(32) // 32 filtres de sortie
                .stride(1, 1)
                .activation(Activation.LEAKYRELU) // utilise la fonction d'activation RELU 
                .build())
            .layer(new BatchNormalization())
             // deuxième couche de convolution
            .layer(new ConvolutionLayer.Builder(3, 3) //filtre 3x3
                .nOut(32) // 32 filtres de sortie
                .stride(1, 1)
                .activation(Activation.LEAKYRELU) // utilise la fonction d'activation RELU
                .build())
            .layer(new BatchNormalization())
             //  couche de pooling
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(1, 1)
                .build())
             // troisieme couche de convolution
            .layer(new ConvolutionLayer.Builder(2, 2) //filtre 2x2
                .nOut(64) // 64 filtres de sortie
                .stride(1, 1)
                .activation(Activation.LEAKYRELU) // utilise la fonction d'activation RELU
                .build())
            .layer(new BatchNormalization())
             // quatrieme couche de convolution
            .layer(new ConvolutionLayer.Builder(2, 2) //filtre 2x2
                .nOut(128) // 128 filtres de sortie
                .stride(1, 1)
                .activation(Activation.LEAKYRELU) // utilise la fonction d'activation RELU
                .build())
            .layer(new BatchNormalization())
             // couche dense
            .layer(new DenseLayer.Builder()
                .nOut(512) // 512 neurones
                .activation(Activation.LEAKYRELU) // utilise la fonction d'activation RELU
                .dropOut(0.4)  // Ajout de dropout pour éviter le surapprentissage
                .build())
            .layer(new DenseLayer.Builder()
                .nOut(256) //256 neurones
                .activation(Activation.LEAKYRELU) // utilise la fonction d'activation RELU
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
        int numExamples = fullDataset.numExamples(); // Calcule la taille du dataset
        int trainSize = (int) (numExamples * trainRatio); // Calcule 80% pour l'entrainement si trainRatio = 0.8
        
        SplitTestAndTrain splitSets = fullDataset.splitTestAndTrain(trainSize); //Divise le dataset en deux parties avec les 80% calculés avant donc 80% entrainement et 20% test
        return new DataSet[]{splitSets.getTrain(), splitSets.getTest()};
    }
    
    
}

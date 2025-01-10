/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

import ProjetChloeTheo.Apprentissage.config_Othello.JeuOthello;
import ProjetChloeTheo.Apprentissage.config_Othello.CoupOthello;
import ProjetChloeTheo.Apprentissage.config_Othello.ResumeResultat;
import ProjetChloeTheo.Apprentissage.config_Othello.ChoixCoup;
import ProjetChloeTheo.Apprentissage.config_Othello.Joueur;
import ProjetChloeTheo.Apprentissage.config_Othello.StatutSituation;
import ProjetChloeTheo.Apprentissage.config_Othello.SituationOthello;
import ProjetChloeTheo.Apprentissage.oracles.OracleStupide;
import ProjetChloeTheo.Apprentissage.oracles.Oracle;
import ProjetChloeTheo.Apprentissage.oracles.OracleCNN;
import ProjetChloeTheo.Apprentissage.oracles.OraclePerceptron;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.nd4j.linalg.dataset.DataSet;

/**
 *
 * @author francois
 */
public class MAIN {
    
    // Ajoutez cette énumération en haut de la classe MAIN pour définir les types de modèles
    public enum ModelType {
        PERCEPTRON,
        CNN
    }
    
    // Méthode modifiée pour créer l'oracle approprié selon le type
    private static Oracle createOracle(Joueur joueur, String modelPath, ModelType modelType) throws IOException {
        switch (modelType) {
            case PERCEPTRON:
                return new OraclePerceptron(joueur, modelPath, true);
            case CNN:
                return new OracleCNN(joueur, modelPath, true);
            default:
                throw new IllegalArgumentException("Type de modèle non supporté");
        }
    }
    
    //Génère une ligne CSV pour une situation donnée
    
    private static void generateUneLigneCSVOfSituations(
            Writer curWriter,
            SituationOthello curSit, double res, int numCoup, int totCoup,
            boolean includeRes, boolean includeNumCoup, boolean includeTotCoup) throws IOException {
        curWriter.append(curSit.toCSV());
        if (includeRes) {
            curWriter.append("," + res);
        }
        if (includeNumCoup) {
            curWriter.append("," + numCoup);
        }
        if (includeTotCoup) {
            curWriter.append("," + totCoup);
        }
        curWriter.append("\n");
    }
    
    
    
    // Génère le CSV pour les situations de jeu
    
    public static void generateCSVOfSituations(         
            Writer outJ1, Writer outJ2,
            JeuOthello jeu, Oracle j1, Oracle j2,
            int nbrParties,
            boolean includeRes, boolean includeNumCoup, boolean includeTotCoup,
            Random rand) throws IOException {
        
        int Quigagne[]= new int[2]; //[0] pour noir, [1] pour blanc
        
        
        for (int i = 0; i < nbrParties; i++) {
            ResumeResultat resj = jeu.partie(j1,ChoixCoup.ORACLE_MEILLEUR,    // LIGNE POUR MODIFIER LES CHOIX COUPS DES ORACLES
                    j2, ChoixCoup.ORACLE_MEILLEUR, false, false, rand,false);  
            SituationOthello curSit = jeu.situationInitiale();
            Writer curOut = outJ1;
            double curRes;
            if (resj.getStatutFinal() == StatutSituation.NOIR_GAGNE) {
                curRes = 1;
                System.out.println("Partie " + (i + 1) + ": Les noirs ont gagné");
                Quigagne[0]++;
            } else if (resj.getStatutFinal() == StatutSituation.BLANC_GAGNE) {
                curRes = 0;
                System.out.println("Partie " + (i + 1) + ": Les blancs ont gagné");
                Quigagne[1]++;
            } else if (resj.getStatutFinal() == StatutSituation.MATCH_NUL) {
                curRes = 0.5;
                System.out.println("Partie " + (i + 1) + ": Match nul");
            } else {
                throw new Error("partie non finie");
            }
            int totCoups = resj.getCoupsJoues().size();
            int numCoup = 0;
            generateUneLigneCSVOfSituations(curOut, curSit, curRes, numCoup, totCoups, includeRes, includeNumCoup, includeTotCoup);  //générer la ligne initiale de la situation init
            Joueur curJoueur = Joueur.NOIR;
            for (CoupOthello curCoup : resj.getCoupsJoues()) {
                curSit = jeu.updateSituation(curSit, curJoueur, curCoup);
                if (curOut == outJ1) {
                    curOut = outJ2;
                } else {
                    curOut = outJ1;
                }
                curRes = 1 - curRes;
                numCoup++;
                generateUneLigneCSVOfSituations(curOut, curSit, curRes, numCoup, totCoups, includeRes, includeNumCoup, includeTotCoup); //générer nouvelle ligne à chaque coup
                curJoueur = curJoueur.adversaire();
            }
        }
        
        // Affichage des pourcentages de victoire
        int totalParties = Quigagne[0] + Quigagne[1];
        double pourcentageNoirs = (double) Quigagne[0] / totalParties * 100;
        double pourcentageBlancs = (double) Quigagne[1] / totalParties * 100;

        System.out.printf("Les noirs ont gagné %d fois (%.2f%%)\n", Quigagne[0], pourcentageNoirs);
        System.out.printf("Les blancs ont gagné %d fois (%.2f%%)\n", Quigagne[1], pourcentageBlancs);
        
    }
    
    // Crée le fichier csv pour une série de parties
    public static void creationPartie(
            File outJ1, File outJ2,
            JeuOthello jeu, Oracle j1, Oracle j2,
            int nbrParties,
            boolean includeRes, boolean includeNumCoup, boolean includeTotCoup,
            Random rand) throws IOException {
        try (FileWriter wJ1 = new FileWriter(outJ1); FileWriter wJ2 = new FileWriter(outJ2)) {
            generateCSVOfSituations(wJ1, wJ2, jeu, j1, j2, nbrParties, includeRes, includeNumCoup, includeTotCoup,rand);
        }
    }
    
    // Test de base avec les oracles stupides
    public static void testAvecOthello(int nbr) {
        try {
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvSansEntrainement"); //attention je crois que sur l'ordi de chloe c'est temp
            //File dir = new File("C:\\temp");
            creationPartie(new File(dir, "noirs" + nbr +".csv"), new File(dir, "blancs"+nbr+".csv"),
                    new JeuOthello(), new OracleStupide(Joueur.NOIR), new OracleStupide(Joueur.BLANC),nbr, true, false, false,new Random());
        } catch (IOException ex){
            throw new Error(ex);
        }
    }
    
    //Test avec les oracles intelligents en séquentielle et création des fichiers csv normal et avec proba
     public static long testAvecOthelloV2(int nbr, String modelPath,String modelPath1) {
         long startTime = System.currentTimeMillis();
         try {
           
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement");
            
            // Création des parties et générations du csv normal
            JeuOthello jeu = new JeuOthello();            
            Oracle j1 = new OracleCNN(Joueur.NOIR, modelPath, true);
            Oracle j2 = new OracleCNN(Joueur.BLANC, modelPath1,true);
            
            System.out.println("Modèles chargés avec succès.");
            System.out.println("\nDébut des parties...");
            
            creationPartie(
                    new File(dir, "noirs" + nbr + "OMCNN-OMCNN2.csv"), 
                    new File(dir, "blancs" + nbr + "OMCNN-OMCNN2.csv"),
                    jeu, j1, j2, nbr, true, false, false, new Random());
            
            System.out.println("Création du fichier csv avec proba ...");
            // Création du fichier csv avec proba
            String inputFile = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement\\noirs" + nbr + "OMCNN-OMCNN2.csv";
            String outputFile = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba" + nbr + "OOMCNN-OMCNN2.csv";
            CsvAvecProba.createCsvProba(inputFile, outputFile);
            
            
        } catch (IOException ex) {
            throw new Error(ex);
        }
        return System.currentTimeMillis() - startTime;
    }
     
   
    
    // test avec parallélisation des parties 
    public static long testAvecOthelloV3(int nbrParties, String modelPath1, String modelPath2, 
                                   ModelType typeJ1, ModelType typeJ2) {
    long startTime = System.currentTimeMillis();// prend le temps de départ
    try {
        File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement");
        String modelDesc = String.format("%s-%s", typeJ1.toString(), typeJ2.toString()); // permet de concaténer les noms des oracles pour mieux se répérer dans les fichiers
        File fileJ1 = new File(dir, "noirs" + nbrParties + modelDesc + ".csv"); // création du fichier pour les pions noirs
        File fileJ2 = new File(dir, "blancs" + nbrParties + modelDesc + ".csv"); // création du fichier pour les pions noirs
        
        // création du jeu
        JeuOthello jeu = new JeuOthello(); 
        // Création des oracles selon leur type Perceptron ou CNN
        Oracle j1 = createOracle(Joueur.NOIR, modelPath1, typeJ1);
        Oracle j2 = createOracle(Joueur.BLANC, modelPath2, typeJ2);
        
        System.out.println("Modèles chargés avec succès.");
        System.out.println("\nDébut des parties...");

        // Création des writers avec synchronisation
        FileWriter wJ1 = new FileWriter(fileJ1); // writer pour le joueur noir
        FileWriter wJ2 = new FileWriter(fileJ2); // writer pour le joueur blanc
        Object writerLock = new Object(); // locker qui nous permettra par la suite de synchroniser l'écriture dans les fichiers csv
        int[] victoires = new int[2]; // Tableau compteur de victoire pour les pions noirs et blancs

        int nbrThreads = Runtime.getRuntime().availableProcessors(); // permet de connaitre le nombre de processeurs utilisable et donc le nombre de thread
        int partiesParThread = nbrParties / nbrThreads; // départage le nombre de partie entre les threads de façon équitable
        
        List<Thread> threads = new ArrayList<>(); // crée une liste de threads
        
        for (int i = 0; i < nbrThreads; i++) {
            final int threadIndex = i;
            Thread t = new Thread(() -> {
                // détermine le début et la fin de toutes les parties faites sur les différents threads
                int debut = threadIndex * partiesParThread;
                int fin = (threadIndex == nbrThreads - 1) ? nbrParties : debut + partiesParThread;
                
                // crée les parties 
                for (int p = debut; p < fin; p++) { 
                    try {
                        ResumeResultat resj = jeu.partie(j1, ChoixCoup.ORACLE_PONDERE,
                                j2, ChoixCoup.ORACLE_PONDERE, false, false, new Random(), false);
                        
                        synchronized (victoires) { // permet de synchronisé le nombre de victoire des parties faites sur les différents threads
                            if (resj.getStatutFinal() == StatutSituation.NOIR_GAGNE) {
                                victoires[0]++;
                                System.out.println("Partie " + (p + 1) + ": Les noirs ont gagné");
                            } else if (resj.getStatutFinal() == StatutSituation.BLANC_GAGNE) {
                                victoires[1]++;
                                System.out.println("Partie " + (p + 1) + ": Les blancs ont gagné");
                            } else {
                                System.out.println("Partie " + (p + 1) + ": Match nul");
                            }
                        }
                        
                        synchronized (writerLock) { // permet d'écrire dnas le csv les parties jouées sur les différents threads de manière synchronisée
                            SituationOthello curSit = jeu.situationInitiale();
                            Writer curOut = wJ1;
                            double curRes;
                            if (resj.getStatutFinal() == StatutSituation.NOIR_GAGNE) {
                                curRes = 1.0;
                            } else if (resj.getStatutFinal() == StatutSituation.BLANC_GAGNE) {
                                curRes = 0.0;
                            } else {
                                curRes = 0.5;
                            }
                            generateUneLigneCSVOfSituations(curOut, curSit, curRes, 0, 
                                    resj.getCoupsJoues().size(), true, false, false);
                            
                            Joueur curJoueur = Joueur.NOIR;
                            for (CoupOthello curCoup : resj.getCoupsJoues()) {
                                curSit = jeu.updateSituation(curSit, curJoueur, curCoup);
                                if (curOut == wJ1){
                                    curOut = wJ2;
                                }else{
                                    curOut = wJ1;
                                }
                                curRes = 1 - curRes;
                                generateUneLigneCSVOfSituations(curOut, curSit, curRes, 0,
                                        resj.getCoupsJoues().size(), true, false, false);
                                curJoueur = curJoueur.adversaire();
                            }
                        }
                    } catch (IOException e) {
                        System.err.println("Erreur lors de l'écriture pour la partie " + p + ": " + e.getMessage());
                    }
                }
            });
            threads.add(t); 
            t.start();
        }

        for (Thread t : threads) {
            t.join();
        }

        wJ1.close();
        wJ2.close();

        int totalParties = victoires[0] + victoires[1];
        double pourcentageNoirs = (double) victoires[0] / totalParties * 100;
        double pourcentageBlancs = (double) victoires[1] / totalParties * 100;
        System.out.printf("Les noirs ont gagné %d fois (%.2f%%)\n", victoires[0], pourcentageNoirs);
        System.out.printf("Les blancs ont gagné %d fois (%.2f%%)\n", victoires[1], pourcentageBlancs);

        System.out.println("Création du fichier csv avec proba ...");
        String inputFile = fileJ1.getPath();
        String outputFile = inputFile.replace("CsvAvecEntrainement", "CsvAvecEntrainementProba")
                                  .replace("noirs", "noirsProba");
        CsvAvecProba.createCsvProba(inputFile, outputFile);
        
    } catch (IOException | InterruptedException ex) {
        throw new Error(ex);
    }
    return System.currentTimeMillis() - startTime; // détermine la fin du temps
}
   

   

    
    public static void comparePerformance(int nbr, String modelPath, String modelPath1, ModelType type1, ModelType type2) {
    System.out.println("Début des tests de performance pour " + nbr + " parties");
    System.out.println("=============================================");

    // Test de la version séquentielle
    System.out.println("\nExécution de la version séquentielle...");
    System.out.println("Configuration: " + type1 + " vs " + type2);
    long seqTime = testAvecOthelloV2(nbr, modelPath, modelPath1);

    // Petit délai pour laisser le système se reposer
    try {
        Thread.sleep(2000);
    } catch (InterruptedException e) {
        e.printStackTrace();
    }

    // Test de la version parallèle
    System.out.println("\nExécution de la version parallèle...");
    System.out.println("Configuration: " + type1 + " vs " + type2);
    long parTime = testAvecOthelloV3(nbr, modelPath, modelPath1, type1, type2);

    // Affichage des résultats
    System.out.println("\nRésultats de la comparaison:");
    System.out.println("=============================================");
    System.out.printf("Temps d'exécution séquentiel: %.2f secondes%n", seqTime / 1000.0);
    System.out.printf("Temps d'exécution parallèle: %.2f secondes%n", parTime / 1000.0);
    System.out.printf("Accélération: %.2fx%n", (double)seqTime / parTime);
    System.out.printf("Amélioration des performances: %.1f%%%n", 
        ((double)(seqTime - parTime) / seqTime) * 100);
}
        
    
  
    
    // Exemple d'utilisation dans le main
    public static void main(String[] args) {
        String modelPathCNN = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn2-model.zip";
        String modelPathPerceptron = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-perceptron2-model.zip";

        // Test CNN vs Perceptron
        //testAvecOthelloV3(500, modelPathCNN, modelPathPerceptron, ModelType.CNN, ModelType.PERCEPTRON);
        // Test CNN vs CNN
        //System.out.println("\n=== Test CNN vs CNN ===");
        //testAvecOthelloV3(500, modelPathCNN, modelPathCNN, ModelType.CNN, ModelType.CNN);

        // Test Perceptron vs CNN avec couleurs inversées
        
        testAvecOthelloV3(300, modelPathPerceptron, modelPathCNN, ModelType.PERCEPTRON, ModelType.CNN);
        

        // Ou Perceptron vs Perceptron
        // testAvecOthelloV3(300, modelPathPerceptron, modelPathPerceptron, ModelType.PERCEPTRON, ModelType.PERCEPTRON);
        
        // Nombre de parties pour le test
        //int nbrParties = 100; // Vous pouvez ajuster ce nombre
        
        // Lancer la comparaison
        //comparePerformance(nbrParties, modelPathCNN, modelPathPerceptron);
    }
    
    

}

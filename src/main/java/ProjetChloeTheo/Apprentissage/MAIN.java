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
import ProjetChloeTheo.Apprentissage.database.DataBaseEnvironment;
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
    
    // Définition des types de modèles
    public enum ModelType {
        PERCEPTRON,
        CNN
    }
    
    // Création des oracles en fonction du type de modèle CNN ou MLP
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
                    j2, ChoixCoup.ALEA, false, false, rand,false);  
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
    
    // Cette méthode permet de jouer des partie entre les oracles stupides en séquentielle et crée le fichier CSV initial
    public static void testAvecOthello(int nbr) {
        try {
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvSansEntrainement"); 
            creationPartie(new File(dir, "noirs" + nbr +".csv"), new File(dir, "blancs"+nbr+".csv"),
                    new JeuOthello(), new OracleStupide(Joueur.NOIR), new OracleStupide(Joueur.BLANC),nbr, true, false, false,new Random());
        } catch (IOException ex){
            throw new Error(ex);
        }
    }
    
     // Cette méthode permet de jouer des partie entre les oracles stupides en séquentielle et crée le fichier CSV initial et le stocké dans la base de donnée
    public static void testAvecOthelloBD(int nbr) {
        try {
            // Connexion à la base de données
            DataBaseEnvironment.connect();
            
            // Chemin du dossier pour les fichiers temporaires
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvSansEntrainement");
            if (!dir.exists()) {
                dir.mkdirs();
            }
            
            // Création des fichiers CSV
            File fileNoirsCSV = new File(dir, "noirs" + nbr + ".csv");
            File fileBlancsCSV = new File(dir, "blancs" + nbr + ".csv");
            
            // Génération des parties et création des CSV
            creationPartie(fileNoirsCSV, fileBlancsCSV,
                    new JeuOthello(), new OracleStupide(Joueur.NOIR), 
                    new OracleStupide(Joueur.BLANC), nbr, true, false, false, new Random());
            
            // Export des fichiers vers la base de données
            System.out.println("Export des fichiers vers la base de données...");
            DataBaseEnvironment.exporterCSVversDatabase(fileNoirsCSV.getAbsolutePath(), "CSVsansEntrainement");
            DataBaseEnvironment.exporterCSVversDatabase(fileBlancsCSV.getAbsolutePath(), "CSVsansEntrainement");
            
            // Fermeture de la connexion
            DataBaseEnvironment.close();
            
            System.out.println("Fichiers exportés avec succès dans la base de données");
        } catch (IOException ex){
            throw new Error(ex);
        }
    }
    
    //Cette méthode permet de jouer un nombre de partie donné avec des oracles intelligents en séquentielle et de créer le fichier csv initial et le fichier csv avec probabilité 
     public static long testAvecOthelloV2(int nbr, String modelPath,String modelPath1) {
         long startTime = System.currentTimeMillis(); // Debut de l'éxécution des parties
         try {
           
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement");
            
            // Création des parties et générations du csv normal
            JeuOthello jeu = new JeuOthello();            
            Oracle j1 = new OracleCNN(Joueur.NOIR, modelPath, true);
            Oracle j2 = new OraclePerceptron(Joueur.BLANC, modelPath1,true);
            
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
        return System.currentTimeMillis() - startTime; // temps mis pour faire le nombre de partie demandée
    }
     
      //Cette méthode permet de jouer un nombre de partie donné avec des oracles intelligents en séquentielle et de créer le fichier csv initial et le fichier csv avec probabilité et de les stocker directement dans la base de donnée
      public static long testAvecOthelloV2BD(int nbr, String modelPath, String modelPath1) {
        long startTime = System.currentTimeMillis();
        try {
            // Connexion à la base de données
            DataBaseEnvironment.connect();
            
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement");
            if (!dir.exists()) {
                dir.mkdirs();
            }
            
            // Création des fichiers CSV
            File fileNoirsCSV = new File(dir, "noirs" + nbr + "OMCNN-OMCNN2.csv");
            File fileBlancsCSV = new File(dir, "blancs" + nbr + "OMCNN-OMCNN2.csv");
            
            // Création des parties et générations du csv normal
            JeuOthello jeu = new JeuOthello();
            Oracle j1 = new OracleCNN(Joueur.NOIR, modelPath, true);
            Oracle j2 = new OraclePerceptron(Joueur.BLANC, modelPath1, true);
            
            System.out.println("Modèles chargés avec succès.");
            System.out.println("\nDébut des parties...");
            
            //joue les parties
            creationPartie(fileNoirsCSV, fileBlancsCSV, jeu, j1, j2, nbr, true, false, false, new Random());
            
            // Export des fichiers vers la base de données
            System.out.println("Export des fichiers vers la base de données...");
            DataBaseEnvironment.exporterCSVversDatabase(fileNoirsCSV.getAbsolutePath(), "CSVavecEntrainement");
            DataBaseEnvironment.exporterCSVversDatabase(fileBlancsCSV.getAbsolutePath(), "CSVavecEntrainement");
            
            System.out.println("Création du fichier csv avec proba ...");
            
            // Création et export du fichier avec probabilités
            String outputFile = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba" + nbr + "OOMCNN-OMCNN2.csv";
            File fileProba = new File(outputFile);
            CsvAvecProba.createCsvProba(fileNoirsCSV.getAbsolutePath(), outputFile);
            
            // Export du fichier proba vers la base de données
            DataBaseEnvironment.exporterCSVversDatabase(fileProba.getAbsolutePath(), "CSVavecEntrainementProba");
            
            // Fermeture de la connexion
            DataBaseEnvironment.close();
            
        } catch (IOException ex) {
            throw new Error(ex);
        }
        return System.currentTimeMillis() - startTime;
    }
     
   
    
    //Cette méthode permet de jouer un nombre de partie donné avec des oracles intelligents en PARALLELE et de créer le fichier csv initial et le fichier csv avec probabilité 
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
                int fin;
                if (threadIndex == nbrThreads - 1) {
                    fin = nbrParties;  // Si c'est le dernier thread, la valeur de fin est nbrParties
                } else {
                    fin = debut + partiesParThread;  // Sinon, fin est égal à debut + partiesParThread
                }
                
                // crée les parties 
                for (int p = debut; p < fin; p++) { 
                    try {
                        ResumeResultat resj = jeu.partie(j1, ChoixCoup.ORACLE_MEILLEUR,
                                j2, ChoixCoup.ALEA, false, false, new Random(), false);
                        
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
                        
                        synchronized (writerLock) { // permet d'écrire dans le csv les parties jouées sur les différents threads de manière synchronisée
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
            threads.add(t); // ajoute le thread à la liste de thread 
            t.start(); // démarre le thread
        }

        for (Thread t : threads) {
            t.join(); // attends la fin de chaque thread
        }
        // fermeture des writers
        wJ1.close();
        wJ2.close();
        
        //calcule des résultats des parties en pourcentage 
        int totalParties = victoires[0] + victoires[1];
        double pourcentageNoirs = (double) victoires[0] / totalParties * 100;
        double pourcentageBlancs = (double) victoires[1] / totalParties * 100;
        System.out.printf("Les noirs ont gagné %d fois (%.2f%%)\n", victoires[0], pourcentageNoirs);
        System.out.printf("Les blancs ont gagné %d fois (%.2f%%)\n", victoires[1], pourcentageBlancs);
        
        //création du fichier csv proba
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
   
    public static long testAvecOthelloV3DB(int nbrParties, String modelPath1, String modelPath2, 
                                   ModelType typeJ1, ModelType typeJ2) {
    long startTime = System.currentTimeMillis();
    try {
        // Connexion à la base de données
        DataBaseEnvironment.connect();
        
        // Création des dossiers pour les fichiers temporaires
        File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        
        // Nom des fichiers avec type de modèle
        String modelDesc = String.format("%s-%s", typeJ1.toString(), typeJ2.toString());
        File fileJ1 = new File(dir, "noirs" + nbrParties + modelDesc + ".csv");
        File fileJ2 = new File(dir, "blancs" + nbrParties + modelDesc + ".csv");
        
        // Création du jeu et des oracles
        JeuOthello jeu = new JeuOthello();
        Oracle j1 = createOracle(Joueur.NOIR, modelPath1, typeJ1);
        Oracle j2 = createOracle(Joueur.BLANC, modelPath2, typeJ2);
        
        System.out.println("Modèles chargés avec succès.");
        System.out.println("\nDébut des parties...");

        // Writers avec synchronisation
        FileWriter wJ1 = new FileWriter(fileJ1);
        FileWriter wJ2 = new FileWriter(fileJ2);
        Object writerLock = new Object();
        int[] victoires = new int[2];

        // Configuration des threads
        int nbrThreads = Runtime.getRuntime().availableProcessors();
        int partiesParThread = nbrParties / nbrThreads;
        List<Thread> threads = new ArrayList<>();
        
        // Création et démarrage des threads
        for (int i = 0; i < nbrThreads; i++) {
            final int threadIndex = i;
            Thread t = new Thread(() -> {
                int debut = threadIndex * partiesParThread;
                int fin = (threadIndex == nbrThreads - 1) ? nbrParties : debut + partiesParThread;
                
                for (int p = debut; p < fin; p++) {
                    try {
                        ResumeResultat resj = jeu.partie(j1, ChoixCoup.ORACLE_MEILLEUR,
                                j2, ChoixCoup.ALEA, false, false, new Random(), false);
                        
                        synchronized (victoires) {
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
                        
                        synchronized (writerLock) {
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
                                curOut = (curOut == wJ1) ? wJ2 : wJ1;
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

        // Attente de la fin des threads
        for (Thread t : threads) {
            t.join();
        }
        
        // Fermeture des writers
        wJ1.close();
        wJ2.close();
        
        // Affichage des résultats
        int totalParties = victoires[0] + victoires[1];
        double pourcentageNoirs = (double) victoires[0] / totalParties * 100;
        double pourcentageBlancs = (double) victoires[1] / totalParties * 100;
        System.out.printf("Les noirs ont gagné %d fois (%.2f%%)\n", victoires[0], pourcentageNoirs);
        System.out.printf("Les blancs ont gagné %d fois (%.2f%%)\n", victoires[1], pourcentageBlancs);
        
        // Export des fichiers CSV vers la base de données
        System.out.println("\nExport des fichiers CSV vers la base de données...");
        DataBaseEnvironment.exporterCSVversDatabase(fileJ1.getAbsolutePath(), "CSVavecEntrainement");
        DataBaseEnvironment.exporterCSVversDatabase(fileJ2.getAbsolutePath(), "CSVavecEntrainement");
        
        // Création et export du fichier avec probabilités
        System.out.println("Création et export du fichier CSV avec probabilités...");
        String outputFile = fileJ1.getPath()
                .replace("CsvAvecEntrainement", "CsvAvecEntrainementProba")
                .replace("noirs", "noirsProba");
        CsvAvecProba.createCsvProba(fileJ1.getPath(), outputFile);
        DataBaseEnvironment.exporterCSVversDatabase(outputFile, "CSVavecEntrainementProba");
        
        // Fermeture de la connexion à la base de données
        DataBaseEnvironment.close();
        System.out.println("Traitement terminé avec succès.");
        
    } catch (IOException | InterruptedException ex) {
        throw new Error(ex);
    }
    return System.currentTimeMillis() - startTime;
}


    // methode qui permet de comparer la rapidité d'execution des parties entre le séquentielle et le parallèle
    public static void comparePerformance(int nbr, String modelPath, String modelPath1, ModelType type1, ModelType type2) {
    System.out.println("Début des tests de performance pour " + nbr + " parties");
    System.out.println("=============================================");

    // Test de la version séquentielle
    System.out.println("\nExécution de la version séquentielle...");
    System.out.println("Configuration: " + type1 + " vs " + type2);
    long seqTime = testAvecOthelloV2(nbr, modelPath, modelPath1);

    // Petit délai 
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
        
        
        // Connection à la base de données
        DataBaseEnvironment.connect();

        // Définir le chemin du dossier de destination
        String cheminDossierCible = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model";

        // Nom du fichier que vous cherchez
        String nomFichier = "othello-cnn4-model.zip";

        // Requête pour trouver l'ID du fichier par son nom
        String query = "SELECT id FROM Model WHERE nom_fichier = '" + nomFichier + "'";
        List<List<Object>> results = DataBaseEnvironment.executeSelectQuery(query);

        if (!results.isEmpty()) {
            int id = ((Number) results.get(0).get(0)).intValue();
            System.out.println("Récupération du fichier: " + nomFichier);
            DataBaseEnvironment.recupererZIPDepuisDatabase(id, cheminDossierCible);
            System.out.println("Fichier récupéré avec succès");
        } else {
            System.out.println("Fichier non trouvé dans la base de données");
        }

        // Fermer la connexion
        DataBaseEnvironment.close();
        
        
        String modelPathCNN = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn4-model.zip";
        String modelPathPerceptron = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-perceptron3-model.zip";
        
        // Test CNN vs Perceptron
        testAvecOthelloV3DB(50, modelPathCNN, modelPathPerceptron, ModelType.CNN, ModelType.PERCEPTRON);
        // Test CNN vs CNN
        //System.out.println("\n=== Test CNN vs CNN ===");
        //testAvecOthelloV3(500, modelPathCNN, modelPathCNN, ModelType.CNN, ModelType.CNN);

        // Test Perceptron vs CNN avec couleurs inversées
        
        //testAvecOthelloV3(200, modelPathPerceptron, modelPathCNN, ModelType.PERCEPTRON, ModelType.CNN);
        

        // Ou Perceptron vs Perceptron
        //testAvecOthelloV3(10000, modelPathPerceptron, modelPathPerceptron, ModelType.PERCEPTRON, ModelType.PERCEPTRON);
        
        // Nombre de parties pour le test
        //int nbrParties = 100; // Vous pouvez ajuster ce nombre
        
        // Lancer la comparaison
        //comparePerformance(nbrParties, modelPathCNN, modelPathPerceptron, ModelType.CNN, ModelType.PERCEPTRON );
        
        //connection base de donnée
        //DataBaseEnvironment.connect();
        
                 
       // Définir le chemin du dossier contenant les fichiers CSV
        //String cheminDossier = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model";
        
        //DataBaseEnvironment.exporterTousLesCSVduDossierVersDB(cheminDossier, "CSVsansEntrainement");
        //DataBaseEnvironment.exporterTousLesZIPduDossierVersDB(cheminDossier, "Model");
       // DataBaseEnvironment.close();
    }
    
    

}

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

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
                    j2, ChoixCoup.ORACLE_PONDERE, false, false, rand,false);  
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
    
     // test avec parallélisation
    public static long testAvecOthelloV3(int nbrParties, String modelPath, String modelPath1) {
        long startTime = System.currentTimeMillis();
        try {
        File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement");
        File fileJ1 = new File(dir, "noirs" + nbrParties + "OMCNN-OMCNN2.csv");
        File fileJ2 = new File(dir, "blancs" + nbrParties + "OMCNN-OMCNN2.csv");
        
        // Création des parties et générations du csv normal
        JeuOthello jeu = new JeuOthello();            
        Oracle j1 = new OracleCNN(Joueur.NOIR, modelPath, true);
        Oracle j2 = new OracleCNN(Joueur.BLANC, modelPath1, true);
        
        System.out.println("Modèles chargés avec succès.");
        System.out.println("\nDébut des parties...");

        // Création des writers avec synchronisation
        FileWriter wJ1 = new FileWriter(fileJ1);
        FileWriter wJ2 = new FileWriter(fileJ2);
        Object writerLock = new Object(); // Pour synchroniser l'écriture
        int[] victoires = new int[2]; // Pour compter les victoires [0]=noir, [1]=blanc

        // Nombre de threads à utiliser
        int nbrThreads = Runtime.getRuntime().availableProcessors();
        int partiesParThread = nbrParties / nbrThreads;
        
        // Création et démarrage des threads
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < nbrThreads; i++) {
            final int threadIndex = i;
            Thread t = new Thread(() -> {
                int debut = threadIndex * partiesParThread;
                int fin = (threadIndex == nbrThreads - 1) ? nbrParties : debut + partiesParThread;
                
                for (int p = debut; p < fin; p++) {
                    try {
                        ResumeResultat resj = jeu.partie(j1, ChoixCoup.ORACLE_MEILLEUR,
                                j2, ChoixCoup.ORACLE_PONDERE, false, false, new Random(), false);
                        
                        // Traitement du résultat
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

                        // Écriture synchronisée dans les fichiers CSV
                        synchronized (writerLock) {
                            SituationOthello curSit = jeu.situationInitiale();
                            Writer curOut = wJ1;
                            double curRes = (resj.getStatutFinal() == StatutSituation.NOIR_GAGNE) ? 1.0 :
                                          (resj.getStatutFinal() == StatutSituation.BLANC_GAGNE) ? 0.0 : 0.5;
                            
                            generateUneLigneCSVOfSituations(curOut, curSit, curRes, 0, 
                                    resj.getCoupsJoues().size(), true, false, false);
                            
                            Joueur curJoueur = Joueur.NOIR;
                            int numCoup = 0;
                            for (CoupOthello curCoup : resj.getCoupsJoues()) {
                                curSit = jeu.updateSituation(curSit, curJoueur, curCoup);
                                curOut = (curOut == wJ1) ? wJ2 : wJ1;
                                curRes = 1 - curRes;
                                numCoup++;
                                generateUneLigneCSVOfSituations(curOut, curSit, curRes, numCoup,
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

        // Attendre que tous les threads terminent
        for (Thread t : threads) {
            t.join();
        }

        // Fermeture des writers
        wJ1.close();
        wJ2.close();

        // Affichage des statistiques finales
        int totalParties = victoires[0] + victoires[1];
        double pourcentageNoirs = (double) victoires[0] / totalParties * 100;
        double pourcentageBlancs = (double) victoires[1] / totalParties * 100;
        System.out.printf("Les noirs ont gagné %d fois (%.2f%%)\n", victoires[0], pourcentageNoirs);
        System.out.printf("Les blancs ont gagné %d fois (%.2f%%)\n", victoires[1], pourcentageBlancs);

        System.out.println("Création du fichier csv avec proba ...");
        // Création du fichier csv avec proba
        String inputFile = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement\\noirs" + nbrParties + "OMCNN-OMCNN2.csv";
        String outputFile = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba" + nbrParties + "OOMCNN-OMCNN2.csv";
        CsvAvecProba.createCsvProba(inputFile, outputFile);
        
    } catch (IOException | InterruptedException ex) {
        throw new Error(ex);
    }
    return System.currentTimeMillis() - startTime;
}
    
    // Méthode pour comparer les performances
    public static void comparePerformance(int nbr, String modelPath, String modelPath1) {
        System.out.println("Début des tests de performance pour " + nbr + " parties");
        System.out.println("=============================================");

        // Test de la version séquentielle
        System.out.println("\nExécution de la version séquentielle...");
        long seqTime = testAvecOthelloV2(nbr, modelPath, modelPath1);
        
        // Petit délai pour laisser le système se reposer
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        // Test de la version parallèle
        System.out.println("\nExécution de la version parallèle...");
        long parTime = testAvecOthelloV3(nbr, modelPath, modelPath1);

        // Affichage des résultats
        System.out.println("\nRésultats de la comparaison:");
        System.out.println("=============================================");
        System.out.printf("Temps d'exécution séquentiel: %.2f secondes%n", seqTime / 1000.0);
        System.out.printf("Temps d'exécution parallèle: %.2f secondes%n", parTime / 1000.0);
        System.out.printf("Accélération: %.2fx%n", (double)seqTime / parTime);
        System.out.printf("Amélioration des performances: %.1f%%%n", 
            ((double)(seqTime - parTime) / seqTime) * 100);
    }
        
    
    public static void main(String[] args){
        //testAvecOthello(10000);
        
        // Chemin du modèle entraîné pour l'OracleIntelligent
        String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn2-model.zip";
        String modelPath1 = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn-model.zip";
        //String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-perceptron-model-OMPER-OPPER.zip";
        
        
        //Lancement du test
        testAvecOthelloV3(200, modelPath,modelPath1);
        
         // Nombre de parties pour le test
        //int nbrParties = 100; // Vous pouvez ajuster ce nombre
        
        // Lancer la comparaison
        //comparePerformance(nbrParties, modelPath, modelPath1);
    }
    
    

}

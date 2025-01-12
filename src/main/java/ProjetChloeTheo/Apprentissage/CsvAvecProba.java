/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
/**
 *
 * @author chloe
 */



public class CsvAvecProba {
    
    // Méthode qui lit le fichier csv et le transforme en une MAP ayant une clé et une valeur pour chaque clé
    private static Map<String, List<Double>> readAndProcessCSV(String filename) throws IOException {
        Map<String, List<Double>> dataMap = new HashMap<>(); // Création d'une MAP qui associe une clé situation (type string) à une probabilité (type double)
        BufferedReader br = new BufferedReader(new FileReader(filename)); // permet d'ouvrir le fichier csv et de le lire ligne par ligne
        String line;

        while ((line = br.readLine()) != null) { 
            // Diviser la ligne en un tableau de chaine de valeur en divisant à chaque virgule
            String[] values = line.split(",");
            
            if(values.length == 65){ // verifie si la ligne contient bien 65 valeurs 
                String key = ""; // crée une chaine vide
                for (int i=0 ; i<64 ; i++){ // parcourt les 64 premiers élements de la liste qui correspondent à la situation
                    key += values[i] + ","; // On concatène ces valeurs à la variable clé 
                }
                double proba = Double.parseDouble(values[64]); // la 65 valeurs est convertie en un double
                dataMap.computeIfAbsent(key, k -> new ArrayList<>()).add(proba); // on ajoute la 65ème valeur à la situation correspondante dans la MAP sous forme de liste si la situation est apparue plusieurs fois
            }
        }
        br.close();
        return dataMap;
    }
    
    //Cette méthode permet d'associer à chaque situation une moyenne des probabilités
    // exemple de focntionnement : si on a une ligne du dataMAP comme ceci (clé : liste de proba)  "0,0,1,-1,0,....,1,0" : [0.5, 1.0, 0.5,0.0] 
    // on obtient dans le nouveau fichier la ligne suivante : 0,0,1,-1,0,....,1,0 0.5
    private static void writeAveragedDataToCSV(Map<String, List<Double>> dataMap, String outputFile) throws IOException {
        FileWriter writer = new FileWriter(outputFile); // crée un writer qui permet d'écrire dans le nouveau fichier

        for (Map.Entry<String, List<Double>> entry : dataMap.entrySet()) { // on itère sur chaque entrée clé-valeur de la MAP
            String key = entry.getKey(); //Récupère la clé donc la situation
            List<Double> probas = entry.getValue(); // Récupère la valeur associé à la situation

            // Calcule la moyenne des 65e valeurs
            double sum = 0; 
            for (double proba : probas) {
                sum += proba;  
            }
            double probaMoyenne = sum / probas.size();
            
            // Écrire les 64 premières valeurs et la moyenne dans le fichier de sortie
            writer.write(key + probaMoyenne + "\n");
        }

        writer.close();
    }
    
    // Méthode pour créer le csv avec proba
    public static void createCsvProba(String inputFilename, String outputFilename) throws IOException {
        try {
            Map<String, List<Double>> dataMap = readAndProcessCSV(inputFilename);
            writeAveragedDataToCSV(dataMap, outputFilename);
            System.out.println("Processing completed. Averaged data written to " + outputFilename);
        } catch (IOException e) {
            System.err.println("An error occurred while processing the CSV files: " + e.getMessage());
            throw e;  
        }
    }
    
     public static void main(String[] args) {
        String inputFilename = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement\\noirs10000PERCEPTRON-PERCEPTRON.csv";
        String outputFilename = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba10000PERCEPTRON-PERCEPTRON.csv";

        try {
            createCsvProba(inputFilename, outputFilename);
        } catch (IOException e) {
            System.err.println("An error occurred while processing the CSV files: " + e.getMessage());
        }
    }
}
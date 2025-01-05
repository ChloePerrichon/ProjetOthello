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
    
    private static Map<String, List<Double>> readAndProcessCSV(String filename) throws IOException {
        Map<String, List<Double>> dataMap = new HashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;

        while ((line = br.readLine()) != null) {
            // Diviser la ligne en valeurs
            String[] values = line.split(",");
            
            if(values.length == 65){
                String key = "";
                for (int i=0 ; i<64 ; i++){
                    key += values[i] + ",";
                }
                double proba = Double.parseDouble(values[64]);
                dataMap.computeIfAbsent(key, k -> new ArrayList<>()).add(proba);
            }
        }
        br.close();
        return dataMap;
    }

    private static void writeAveragedDataToCSV(Map<String, List<Double>> dataMap, String outputFile) throws IOException {
        FileWriter writer = new FileWriter(outputFile);

        for (Map.Entry<String, List<Double>> entry : dataMap.entrySet()) {
            String key = entry.getKey();
            List<Double> probas = entry.getValue();

            // Calculer la moyenne des 65e valeurs
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
        String inputFilename = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement\\noirs10000OMPER-OPPER.csv";
        String outputFilename = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba10000OMPER-OPPER.csv";

        try {
            createCsvProba(inputFilename, outputFilename);
        } catch (IOException e) {
            System.err.println("An error occurred while processing the CSV files: " + e.getMessage());
        }
    }
}
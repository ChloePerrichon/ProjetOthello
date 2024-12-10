/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

import au.com.bytecode.opencsv.CSVWriter;
import com.opencsv.CSVReader;
import com.opencsv.ICSVWriter;
import com.opencsv.exceptions.CsvValidationException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
/**
 *
 * @author chloe
 */
public class CsvAvecProba {
    private static final String INPUT_CSV = "C:\\temp\\noirs8000.csv";
    private static final String OUTPUT_CSV = "C:\\temp\\noirs8000_with_win_prob.csv";

    public static void main(String[] args) throws CsvValidationException {
        try {
            Map<String, int[]> positionStats = readAndProcessCSV(INPUT_CSV);
            writeNewCSV(OUTPUT_CSV, positionStats);
            System.out.println("Nouveau fichier CSV créé : " + OUTPUT_CSV);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Map<String, int[]> readAndProcessCSV(String filePath) throws IOException, CsvValidationException {
        Map<String, int[]> positionStats = new HashMap<>();

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            String[] nextLine;
            while ((nextLine = reader.readNext()) != null) {
                if (nextLine.length < 67) {
                    System.err.println("Ligne ignorée: nombre incorrect de colonnes");
                    continue;
                }

                StringBuilder positionKeyBuilder = new StringBuilder();
                for (int i = 0; i < 64; i++) {
                    positionKeyBuilder.append(nextLine[i]).append(",");
                }
                String positionKey = positionKeyBuilder.toString();

                int resultPartie = Integer.parseInt(nextLine[65]);

                positionStats.putIfAbsent(positionKey, new int[2]);
                int[] stats = positionStats.get(positionKey);
                stats[0] += resultPartie; // Victoires
                stats[1] += 1; // Parties jouées
            }
        }

        return positionStats;
    }

    private static void writeNewCSV(String filePath, Map<String, int[]> positionStats) throws IOException {
    // Crée un writer sans guillemets
        try (CSVWriter writer = new CSVWriter(new FileWriter(filePath), CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER)) {
            String[] header = new String[65];
            for (int i = 0; i < 64; i++) {
                header[i] = "feature_" + (i + 1);
            }
            header[64] = "win_prob";
            writer.writeNext(header);

            for (Map.Entry<String, int[]> entry : positionStats.entrySet()) {
                String positionKey = entry.getKey();
                int[] stats = entry.getValue();
                double winProb = (double) stats[0] / stats[1];

                String[] row = new String[65];
                String[] positionValues = positionKey.split(",");
                System.arraycopy(positionValues, 0, row, 0, 64);
                row[64] = String.valueOf(winProb);

                writer.writeNext(row);
            }
        }
    }

}
    


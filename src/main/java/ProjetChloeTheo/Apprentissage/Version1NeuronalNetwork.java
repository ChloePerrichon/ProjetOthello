/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */


package ProjetChloeTheo.Apprentissage;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
 /*
 * @author chloe
 */
public class Version1NeuronalNetwork {
    
    public static DataSet createDataset(String csvFilePath) throws IOException {
        List<INDArray> inputList = new ArrayList<>();
        List<INDArray> outputList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length != 68) {
                    throw new IllegalArgumentException("Each row in the CSV file must have 68 columns");
                }

                // Create input INDArray with 64 values
                INDArray input = Nd4j.zeros(1, 64);
                for (int i = 0; i < 64; i++) {
                    input.putScalar(i, Double.parseDouble(values[i]));
                }

                // Create output  INDArray with 1 value
                INDArray output = Nd4j.zeros(1, 1);
                output.putScalar(0, Double.parseDouble(values[64]));

                inputList.add(input);
                outputList.add(output);
            }
        }

        // Stack all input and output INDArrays into single INDArrays
        INDArray input = Nd4j.vstack(inputList);
        INDArray output = Nd4j.vstack(outputList);

        DataSet dataset = new DataSet(input, output);

        // Optional: Normalize the dataset
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataset);
        normalizer.transform(dataset);

        return dataset;
    }

    public static void main(String[] args) {
        try {
            String csvFilePath = "\"C:\\temp\\noirs8000.csv\""; // Path to your CSV file

            DataSet dataset = createDataset(csvFilePath);
            System.out.println(dataset);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
   }
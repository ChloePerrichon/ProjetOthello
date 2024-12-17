/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.testParallel;

import java.util.ArrayList;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author toliveiragaspa01
 */
public class Entraineur implements Runnable {

    private static int TAILLE_BLOC = 1000;
    
    private LinkedBlockingQueue<Double> datas;
    private ArrayList<Double> buffer;

    public Entraineur(LinkedBlockingQueue<Double> datas) {
        this.datas = datas;
        this.buffer = new ArrayList<>(TAILLE_BLOC);
    }

    @Override
    public void run() {
        int lu = 0;
        while (true) {
            try {
                this.buffer.add(datas.poll(Long.MAX_VALUE, TimeUnit.MILLISECONDS));
                lu ++;
                if (lu == TAILLE_BLOC) {
                    double somme = buffer.stream().reduce(0.0, (a,b) -> a + b);
                    System.out.println("somme : " + somme);
                    lu = 0;
                    buffer.clear();
                }
            } catch (InterruptedException ex) {
                throw new Error("no interrupt expected",ex);
            }
        }
    }

}

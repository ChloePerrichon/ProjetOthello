/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.testParallel;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author toliveiragaspa01
 */
public class Partie implements Runnable{
    
    private LinkedBlockingQueue<Double> datas;

    public Partie(LinkedBlockingQueue<Double> datas) {
        this.datas = datas;
    }
    
    

    @Override
    public void run() {
        while(true) {
            try {
                datas.offer(Math.random(), Long.MAX_VALUE, TimeUnit.MILLISECONDS);
                Thread.sleep(10);
            } catch (InterruptedException ex) {
                throw new Error("no inter",ex);
            }
        }
    }
    
    
    
}

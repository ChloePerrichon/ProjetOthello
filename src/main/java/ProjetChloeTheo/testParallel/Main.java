/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.testParallel;

import java.util.concurrent.LinkedBlockingQueue;

/**
 *
 * @author toliveiragaspa01
 */
public class Main {
    
    public static void main(String[] args) {
        LinkedBlockingQueue partage = new LinkedBlockingQueue(10000);
        Entraineur e = new Entraineur(partage);
        Partie p1 = new Partie(partage);
        Partie p2 = new Partie(partage);
        new Thread(e).start();
        new Thread(p1).start();
        new Thread(p2).start();
//        Runtime.getRuntime().availableProcessors()
    }
    
}

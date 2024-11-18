/*
Copyright 2000-2014 Francois de Bertrand de Beuvron

This file is part of CoursBeuvron.

CoursBeuvron is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CoursBeuvron is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CoursBeuvron.  If not, see <http://www.gnu.org/licenses/>.
 */
package ProjetChloeTheo.sockets.chat.texteBasic;

import ProjetChloeTheo.utils.ThreadUtils;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.charset.Charset;

/**
 *
 * @author francois
 */
public class Consumer {

    // normalement pas une bonne idée : le port n'est peut-être pas dispo

    /** todoDoc. */
    public static final int FIXEDPORT = 55555;
    
    /** todoDoc. */
    public static final long DELAY_IN_MS = 5;

    /** todoDoc. */
    public static void start() {
        try (Socket sockIn = new Socket(InetAddress.getLoopbackAddress(), FIXEDPORT)) {
            try (BufferedReader bin = new BufferedReader(new InputStreamReader(sockIn.getInputStream(), Charset.forName("UTF8")))) {
                String line;
                while((line = bin.readLine()) != null) {
                    System.out.println(line.substring(0, 20) + "...");
                    ThreadUtils.sleepSimple(DELAY_IN_MS);
                }
            }

        } catch (IOException ex) {
            throw new Error(ex);
        }
    }
    
    /**
     *
     * @param args
     */
    public static void main(String[] args) {
        start();
    }

}

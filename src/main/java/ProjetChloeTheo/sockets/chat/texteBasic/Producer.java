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

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.Charset;

/**
 *
 * @author francois
 */
public class Producer {

    // normalement pas une bonne idée : le port n'est peut-être pas dispo

    /** todoDoc. */
    public static final int FIXEDPORT = 55555;

    /** todoDoc. */
    public static final int LINE_SIZE = 1000;

    private static String mult(String s, int nbr) {
        StringBuilder res = new StringBuilder(nbr * s.length());
        for (int i = 0; i < nbr; i++) {
            res.append(s);
        }
        return res.toString();
    }

    /** todoDoc. */
    public static void start() {
        InetAddress addr;
        try (ServerSocket serveur = new ServerSocket(FIXEDPORT, 1, InetAddress.getLoopbackAddress())) {
            System.out.println("Socket waiting at :");
            System.out.println("adress : " + serveur.getInetAddress().getHostAddress());
            System.out.println("port : " + serveur.getLocalPort());
            Socket inSock = serveur.accept();
            System.out.println("client : " + inSock + "\n");
            try (Writer buf = new OutputStreamWriter(inSock.getOutputStream(), Charset.forName("UTF8"))) {
                long i = 0;
                String longMess = mult("A", LINE_SIZE);
                while (true) {
                    buf.append(i + " : " + longMess + "\n");
                    System.out.println("envoye a " + inSock.getPort() + " : " + i);
                    i++;
                }
            } catch (IOException ex) {
                throw new Error(ex);
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

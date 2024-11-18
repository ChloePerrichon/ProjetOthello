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
package ProjetChloeTheo.sockets.chat.texteChat;

import ProjetChloeTheo.sockets.INetAdressUtil;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * launch in term in ubuntu : /home/francois/FPrograms/openjdk11/bin/java
 * -classpath
 * /media/veracrypt1/beuvron/dev/java/maven/beuvron/cours/target/cours-1.0-SNAPSHOT.jar
 * fr.insa.beuvron.cours.multiTache.sockets.prodConsoTexte.texteChat.ServerChatText
 *
 * @author francois
 */
public class ServerChatText {

    private ReentrantReadWriteLock servingLock = new ReentrantReadWriteLock(true);
    private List<ThreadSendOne> serving;
    private ThreadConnect connect;

    /** todoDoc. */
    public ServerChatText() {
        this.serving = new ArrayList<>();
        this.connect = this.new ThreadConnect();
    }

    /** todoDoc. */
    public void start() {
        this.connect.start();
    }

    /**
     *
     * @param mess
     */
    public void resendAll(String mess) {
        this.servingLock.readLock().lock();
        try {
            this.serving.forEach((ts) -> ts.addLine(mess));
        } finally {
            this.servingLock.readLock().unlock();
        }
        this.serving.forEach((ts) -> ts.interrupt());
    }

    /** todoDoc. */
    public class ThreadConnect extends Thread {

        private boolean running = false;
        private String hostAdress;
        private int localPort;

        /** todoDoc. */
        @Override
        public void run() {

            InetAddress addr;
            try {
                addr = INetAdressUtil.premiereAdresseNonLoopback();
            } catch (SocketException ex) {
                throw new Error(ex);
            }
            try (ServerSocket serveur = new ServerSocket(0, 1, addr)) {
                System.out.println("Socket serveur waiting at :");
                this.hostAdress = serveur.getInetAddress().getHostAddress();
                System.out.println("adress : " + this.hostAdress);
                this.localPort = serveur.getLocalPort();
                System.out.println("port : " + this.localPort);
                this.running = true;
                while (true) {
                    Socket inSock = serveur.accept();
                    System.out.println("client : " + inSock + "\n");
                    ThreadSendOne ts = new ThreadSendOne(inSock);
                    servingLock.writeLock().lock();
                    try {
                        ServerChatText.this.serving.add(ts);
                        ts.start();
                        ThreadListenOne tlo = ServerChatText.this.new ThreadListenOne(inSock);
                        tlo.start();
                    } finally {
                        servingLock.writeLock().unlock();
                    }
                }
            } catch (IOException ex) {
                throw new Error(ex);
            }

        }

        /**
         * @return the hostAdress
         */
        public String getHostAdress() {
            if (!this.running) {
                throw new Error("server not running yet");
            }
            return hostAdress;
        }

        /**
         * @return the localPort
         */
        public int getLocalPort() {
            if (!this.running) {
                throw new Error("server not running yet");
            }
            return localPort;
        }

        /**
         * @return the running
         */
        public boolean isRunning() {
            return running;
        }

    }

    /** todoDoc. */
    public static class ThreadSendOne extends Thread {

        private Socket inSock;
        private ConcurrentLinkedQueue<String> toSend;

        /**
         *
         * @param inSock
         */
        public ThreadSendOne(Socket inSock) {
            this.inSock = inSock;
            this.toSend = new ConcurrentLinkedQueue<>();
        }

        /**
         *
         * @param line
         */
        public void addLine(String line) {
            this.toSend.add(line);
        }

        /** todoDoc. */
        @Override
        public void run() {
            try (Writer buf = new OutputStreamWriter(inSock.getOutputStream(), Charset.forName("UTF8"))) {
                while (true) {
                    String send; 
                    while ((send = this.toSend.poll()) != null) {
                        System.out.println(this + " sending : " + send);
                        buf.append(send + "\n");
                        buf.flush();
                    }
                    try {
                        System.out.println(this.inSock + " : Sleeping");
                        Thread.sleep(Long.MAX_VALUE);
                    } catch (InterruptedException ex) {
                        System.out.println(this.inSock + " : interrupted sleep");
                        // normal : new message available
                    }

                }
//                    buf.append(i + " : " + this.longMess + "\n");
//                    System.out.println("envoye a " + inSock.getPort() + " : " + i);
//                    i++;
//                }
            } catch (IOException ex) {
                throw new Error(ex);
            } finally {
                if (this.inSock != null) {
                    try {
                        this.inSock.close();
                    } catch (IOException ex) {
                    }
                }
            }

        }
    }

    /** todoDoc. */
    public class ThreadListenOne extends Thread {

        private Socket inSock;
        private String name;

        /**
         *
         * @param inSock
         */
        public ThreadListenOne(Socket inSock) {
            this.inSock = inSock;
        }

        /** todoDoc. */
        @Override
        public void run() {
            try (BufferedReader bufin = new BufferedReader(new InputStreamReader(inSock.getInputStream(), Charset.forName("UTF8")))) {
                // la première ligne envoyee est le nom
                String lineRead;
                if ((lineRead = bufin.readLine()) != null) {
                    this.name = lineRead;

                }
                while (true) {
                    while ((lineRead = bufin.readLine()) != null) {
                        System.out.println(this.inSock + " has send : " + lineRead);
                        ServerChatText.this.resendAll(this.name + " : " +lineRead);
                    }
                }
            } catch (IOException ex) {
                throw new Error(ex);
            } finally {
                if (this.inSock != null) {
                    try {
                        this.inSock.close();
                    } catch (IOException ex) {
                    }
                }
            }

        }
    }

    /**
     *
     * @param args
     */
    public static void main(String[] args) {
        ServerChatText s = new ServerChatText();
        s.start();
    }

}

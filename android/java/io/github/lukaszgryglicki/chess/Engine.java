package io.github.lukaszgryglicki.chess;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * The game itself: chess.go running as a child process. go/frontend.go
 * describes the lines of text exchanged with it.
 */
final class Engine {
    interface Listener {
        /** A line the game wrote; called on the main thread. */
        void onLine(Engine e, String line);

        /** The process is gone; called on the main thread. */
        void onExit(Engine e);
    }

    private static final int SIGCONT = 18, SIGSTOP = 19;
    private static final String END = new String("end"); // told apart from any line by being this object

    private final Process proc;
    private final LinkedBlockingQueue<String> out = new LinkedBlockingQueue<>();
    private int pid = -1;
    private boolean paused;

    /** Starts "libchess.so android args..."; fen is a testing aid, usually empty. */
    Engine(Context c, List<String> args, String fen, final Listener l) throws IOException {
        List<String> cmd = new ArrayList<>();
        // the one place Android lets an app run a program of its own from
        cmd.add(c.getApplicationInfo().nativeLibraryDir + "/libchess.so");
        cmd.add("android");
        cmd.addAll(args);
        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true).directory(c.getCacheDir());
        if (!fen.isEmpty()) {
            pb.environment().put("CHESS_FEN", fen);
        }
        proc = pb.start();
        final Handler main = new Handler(Looper.getMainLooper());
        new Thread(() -> {
            try (BufferedReader r = new BufferedReader(
                    new InputStreamReader(proc.getInputStream(), "UTF-8"), 1 << 16)) {
                for (String s; (s = r.readLine()) != null; ) {
                    final String line = s;
                    main.post(() -> l.onLine(Engine.this, line));
                }
            } catch (IOException e) {
                // the process is gone, which is reported next
            }
            out.offer(END);
            main.post(() -> l.onExit(Engine.this));
        }, "chess-read").start();
        new Thread(() -> {
            try (OutputStream o = proc.getOutputStream()) {
                for (String s; (s = out.take()) != END; ) {
                    o.write((s + "\n").getBytes("UTF-8"));
                    o.flush();
                }
            } catch (IOException | InterruptedException e) {
                // nothing more can be said to it
            }
        }, "chess-write").start();
    }

    void send(String line) {
        out.offer(line);
    }

    /** The process id, which the game reports in its first line. */
    void setPid(int p) {
        pid = p;
        if (paused) {
            android.os.Process.sendSignal(pid, SIGSTOP);
        }
    }

    /** Stops the process where it is: no thinking while nobody looks. */
    void pause() {
        paused = true;
        if (pid > 0) {
            android.os.Process.sendSignal(pid, SIGSTOP);
        }
    }

    void resume() {
        paused = false;
        if (pid > 0) {
            android.os.Process.sendSignal(pid, SIGCONT);
        }
    }

    void kill() {
        out.offer(END);
        proc.destroyForcibly();
    }
}
